
#!/usr/bin/env python3
"""build_patent_features.py

Purpose
-------
Aggregate J-PlatPat-exported patent CSVs (often split into multiple files per company)
under `data/source/patent_counts/` into a single company-level feature table.

Key constraints handled
----------------------
- Some companies have multiple CSVs due to export row limits; all are concatenated.
- Some folders/files like `test` (no extension) or `test.csv` may exist; they are skipped.
- The script must not stop on parse errors; it logs warnings and continues.
- Progress must be visible (which company out of how many).

Outputs
-------
Creates a run folder under `data/features/patents/` and writes:
- patent_features_company.csv : one row per company with the 7 requested metrics
- run_summary.json           : basic processing summary (counts, errors)

Usage
-----
python3 scripts/phase3/patents/build_patent_features.py \
  --input_dir /Users/shou/hobby/CPX/nikkei-stock/data/source/patent_counts \
  --out_dir   /Users/shou/hobby/CPX/nikkei-stock/data/features/patents

Optional
--------
--asof YYYY-MM-DD : 기준일 (default: today). Used for "last 5 years" windows.
--write_by_company : also write per-company feature CSVs under run_dir/by_company/

Notes
-----
This script intentionally avoids ML. It computes reproducible, human-auditable
metrics only from the J-PlatPat export columns.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Configuration / heuristics
# ----------------------------
SKIP_BASENAMES = {
    "test",
    "test.csv",
    ".ds_store",
}

# Keywords to identify maintenance / annuity events
MAINTENANCE_KEYWORDS = (
    "年金",
    "維持",
)

# Keywords to identify universities (co-application proxy)
UNIVERSITY_KEYWORDS = (
    "大学",
    "国立大学法人",
    "公立大学法人",
    "学校法人",
    "University",
)


@dataclass
class CompanyBundle:
    company: str
    files: List[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build company-level patent feature table from J-PlatPat CSV exports")
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/source/patent_counts"),
        help="Root directory containing J-PlatPat exported CSVs (may include subfolders).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/features/patents"),
        help="Base output directory. A run folder will be created under this.",
    )
    p.add_argument(
        "--asof",
        type=str,
        default=None,
        help="As-of date in YYYY-MM-DD for 'last 5 years' window (default: today).",
    )
    p.add_argument(
        "--write_by_company",
        action="store_true",
        help="Also write per-company feature CSVs under run_dir/by_company/ (debug/QA).",
    )
    return p.parse_args()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_company_name(name: str) -> str:
    """Normalize company name for consistent joining.

    - trims whitespace
    - removes common suffixes/markers from filenames
    - does NOT remove '株式会社' by default because fromnotion uses official names;
      instead we keep base name as-is and rely on later mapping if needed.

    For folder-derived names, this is usually already clean.
    """
    s = str(name).strip()
    s = s.replace("　", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Remove typical file suffixes (but keep legal suffix like 株式会社 if it is part of folder name)
    s = re.sub(r"(_?国内文献|_?国内特許|_?文献)$", "", s)
    s = re.sub(r"[_\-\s]*\(?\d+\)?$", "", s)  # trailing _1, -2, (3)
    s = s.strip(" _-()（）")
    return s


def _infer_company_from_path(csv_path: Path, input_root: Path) -> str:
    """Infer company name.

    Priority:
    1) If file is inside a direct subfolder (e.g., patent_counts/戸田建設/*.csv), use folder name.
    2) Otherwise infer from filename (before first underscore / '（' etc.).
    """
    try:
        rel = csv_path.relative_to(input_root)
    except ValueError:
        rel = csv_path

    if len(rel.parts) >= 2:
        # Use the top folder name under input_root
        folder = rel.parts[0]
        if folder and folder.lower() not in SKIP_BASENAMES:
            return _normalize_company_name(folder)

    # Fallback to filename parsing
    stem = csv_path.stem
    # Remove common Japanese parentheses blocks first
    stem = re.split(r"[（(]", stem)[0]
    # Remove after first underscore
    stem = stem.split("_")[0]
    return _normalize_company_name(stem)


def _is_skippable_csv(path: Path) -> bool:
    bn = path.name.lower()
    if bn in SKIP_BASENAMES:
        return True
    if bn.startswith("."):
        return True
    if "test" == path.stem.lower() or path.stem.lower().endswith("_test"):
        return True
    return False


def _find_csvs(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    return sorted([p for p in input_dir.rglob("*.csv") if p.is_file()])


def _bundle_by_company(csv_paths: Iterable[Path], input_root: Path) -> List[CompanyBundle]:
    buckets: Dict[str, List[Path]] = {}
    for p in csv_paths:
        if _is_skippable_csv(p):
            continue
        company = _infer_company_from_path(p, input_root)
        buckets.setdefault(company, []).append(p)
    bundles = [CompanyBundle(company=k, files=sorted(v)) for k, v in buckets.items()]
    bundles.sort(key=lambda b: b.company)
    return bundles


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV robustly.

    - Try utf-8-sig then cp932.
    - Never throws; returns None on failure.
    """
    encodings = ["utf-8-sig", "utf-8", "cp932"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    print(f"[WARN] Failed to read CSV: {path} | last_error={last_err}")
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse Japanese date-like strings into datetime64.

    Accepts formats like:
    - 2023-01-02
    - 2023/1/2
    - 2023.1.2
    - 2023年1月2日
    """
    s = series.astype(str)
    s = s.replace({"nan": "", "None": ""})
    s = s.str.strip()
    # Normalize separators
    s = s.str.replace("年", "-", regex=False)
    s = s.str.replace("月", "-", regex=False)
    s = s.str.replace("日", "", regex=False)
    s = s.str.replace("/", "-", regex=False)
    s = s.str.replace(".", "-", regex=False)
    return pd.to_datetime(s, errors="coerce")


def _split_tokens(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).strip()
    if not t or t.lower() == "nan":
        return []
    # Common separators in FI lists
    parts = re.split(r"[\s,，、;；|]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def compute_features(df: pd.DataFrame, asof: date) -> Dict[str, object]:
    """Compute the 7 requested metrics from concatenated patent rows."""

    total = int(len(df))

    # 出願日
    apps_5y = 0
    if "出願日" in df.columns:
        dt = _parse_date_series(df["出願日"])
        start_year = asof.year - 4  # inclusive window: asof.year-4 .. asof.year
        apps_5y = int((dt.dt.year >= start_year).sum())

    # 登録番号
    grant_count = 0
    if "登録番号" in df.columns:
        grant_count = int(df["登録番号"].notna().sum())
    grant_rate = float(grant_count / total) if total > 0 else 0.0

    # ステージ: "特許 有効"
    valid_count = 0
    if "ステージ" in df.columns:
        stage = df["ステージ"].astype(str)
        valid_count = int(stage.str.contains(r"特許\s*有効", regex=True, na=False).sum())
    valid_rate = float(valid_count / total) if total > 0 else 0.0

    # イベント詳細: maintenance keywords
    maint_count = 0
    if "イベント詳細" in df.columns:
        ev = df["イベント詳細"].astype(str)
        patt = "|".join(map(re.escape, MAINTENANCE_KEYWORDS))
        maint_count = int(ev.str.contains(patt, regex=True, na=False).sum())
    maint_rate = float(maint_count / total) if total > 0 else 0.0

    # 共同出願比率: 出願人/権利者に university or multiple entities
    coapp_count = 0
    applicant_cols = [c for c in ["出願人/権利者", "出願人", "権利者"] if c in df.columns]
    if applicant_cols:
        col = applicant_cols[0]
        ap = df[col].astype(str).fillna("")
        has_univ = ap.str.contains("|".join(map(re.escape, UNIVERSITY_KEYWORDS)), regex=True, na=False)
        # multiple entities if separators indicate more than 1
        multi = ap.str.contains(r"[,，、]", regex=True, na=False)
        coapp_count = int((has_univ | multi).sum())
    coapp_rate = float(coapp_count / total) if total > 0 else 0.0

    # FIユニーク数
    fi_unique = 0
    if "FI" in df.columns:
        tokens: List[str] = []
        for v in df["FI"].tolist():
            tokens.extend(_split_tokens(v))
        fi_unique = int(len(set(tokens)))

    return {
        "doc_count_total": total,
        "apps_count_5y": apps_5y,
        "grant_count": grant_count,
        "grant_rate": grant_rate,
        "valid_patent_count": valid_count,
        "valid_patent_rate": valid_rate,
        "maintenance_event_count": maint_count,
        "maintenance_event_rate": maint_rate,
        "coapp_count": coapp_count,
        "coapp_rate": coapp_rate,
        "fi_unique_count": fi_unique,
    }


def main() -> int:
    args = parse_args()

    input_dir: Path = args.input_dir
    out_base: Path = args.out_dir

    if args.asof:
        asof_dt = datetime.strptime(args.asof, "%Y-%m-%d").date()
    else:
        asof_dt = date.today()

    csvs = _find_csvs(input_dir)
    print(f"[INFO] Found CSV files: {len(csvs)} under {input_dir}")

    bundles = _bundle_by_company(csvs, input_dir)
    print(f"[INFO] Inferred companies: {len(bundles)}")

    run_dir = out_base / f"patent_features_v1_{_now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    by_company_dir = run_dir / "by_company"
    if args.write_by_company:
        by_company_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    skipped_files: List[str] = []
    failed_files: List[str] = []

    # Track skipped patterns explicitly (for transparency)
    for p in csvs:
        if _is_skippable_csv(p):
            skipped_files.append(str(p))

    total_companies = len(bundles)
    for idx, bundle in enumerate(bundles, start=1):
        # Read and concat all CSVs for the company
        dfs: List[pd.DataFrame] = []
        for f in bundle.files:
            df = _safe_read_csv(f)
            if df is None:
                failed_files.append(str(f))
                continue
            dfs.append(df)

        if not dfs:
            print(f"[WARN] [{idx:02d}/{total_companies}] {bundle.company}: no readable CSVs (skipped)")
            continue

        merged = pd.concat(dfs, ignore_index=True)
        feats = compute_features(merged, asof=asof_dt)

        row = {
            "company_name": bundle.company,
            "source_files": "|".join([str(p) for p in bundle.files]),
            "asof": asof_dt.isoformat(),
            **feats,
        }
        rows.append(row)

        print(
            f"[OK] [{idx:02d}/{total_companies}] {bundle.company}: "
            f"files={len(bundle.files)} rows={feats['doc_count_total']} "
            f"apps5y={feats['apps_count_5y']} grant_rate={feats['grant_rate']:.3f} "
            f"valid_rate={feats['valid_patent_rate']:.3f} maint_rate={feats['maintenance_event_rate']:.3f} "
            f"coapp_rate={feats['coapp_rate']:.3f} fi_unique={feats['fi_unique_count']}"
        )

        if args.write_by_company:
            out_one = by_company_dir / f"{bundle.company}_features.csv"
            pd.DataFrame([row]).to_csv(out_one, index=False)

    # Write outputs
    out_csv = run_dir / "patent_features_company.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(run_dir),
        "asof": asof_dt.isoformat(),
        "csv_files_found": len(csvs),
        "companies_inferred": len(bundles),
        "companies_output": len(rows),
        "skipped_files": skipped_files[:200],
        "skipped_files_count": len(skipped_files),
        "failed_files": failed_files[:200],
        "failed_files_count": len(failed_files),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {out_csv}")
    print(f"[DONE] summary: {run_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())