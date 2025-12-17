#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build B: climate commitment flags (GX / TCFD / SBTi) keyed by ticker.

Inputs (expected):
- src/data_collection/processed/universe_with_gx_flag.csv
    columns: ticker, is_gx, (optional company_name, company_name_norm)
- src/data_collection/processed/jpx_company_ticker_map.csv
    columns: ticker, company_name, company_name_norm (at least ticker + company_name_norm)
- src/data_collection/processed/jpx_ticker_english_name.csv
    columns: ticker, yfinance_long_name, yfinance_short_name, english_name_norm
- src/data_collection/raw/TCFDcompanies.csv
    columns: company_name-like field (varies)
- src/data_collection/raw/SBTis_Target_Dashboard.csv
    columns: company_name-like field (varies)

Outputs:
- src/data_collection/processed/climate_commitment_flags.csv
    columns: ticker, is_gx, tcfd_flag, sbti_flag
- src/data_collection/processed/climate_commitment_flags_unmatched.csv
    unmatched company rows (for manual review)
"""

from __future__ import annotations

import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional, List
import warnings

import pandas as pd


# -----------------------------
# Normalization (JP company names)
# -----------------------------
_CORP_SUFFIX_PATTERNS = [
    r"株式会社",
    r"（株）",
    r"\(株\)",
    r"有限会社",
    r"合同会社",
    r"Inc\.?",
    r"Incorporated",
    r"Corp\.?",
    r"Corporation",
    r"Co\.?,?\s*Ltd\.?",
    r"Limited",
    r"Ltd\.?",
    r"PLC",
    r"Holdings?",
    r"ホールディングス",
]

_PUNCT = r"[・･·•\.\,，、。/／\-\–\—_＿\(\)（）\[\]【】{}<>「」『』“”\"'’\s]"


def normalize_company_name(s: str) -> str:
    """
    Normalize company name for matching:
    - Unicode NFKC (full/half width normalize)
    - Uppercase latin
    - Remove common corporate suffixes (株式会社 etc.)
    - Remove punctuation/spaces
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.upper()

    # Remove suffix words (repeat until stable)
    prev = None
    while prev != s:
        prev = s
        for pat in _CORP_SUFFIX_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # Remove punctuation/spaces
    s = re.sub(_PUNCT, "", s)
    return s


# -----------------------------
# Normalization (EN company names)
# -----------------------------
_EN_SUFFIXES = [
    "INC",
    "CORPORATION",
    "CORP",
    "CO",
    "LTD",
    "LIMITED",
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "PLC",
    "AG",
    "SA",
    "NV",
]

_EN_PUNCT = r"[\\s\\.,'\"/&\\-–—_()\\[\\]{}]"

_COMMITMENT_WEIGHT_GX = 30
_COMMITMENT_WEIGHT_TCFD = 20
_COMMITMENT_WEIGHT_SBTI = 30


def normalize_english_name(name: str) -> str:
    """Normalize English company name for matching across datasets and yfinance.

    Steps (English-targeted; do not reuse JP normalization):
    1) Uppercase
    2) Remove punctuation/spaces
    3) Strip common corporate suffixes
    """
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""

    s = s.upper()
    s = re.sub(_EN_PUNCT, "", s)

    prev = None
    while prev != s:
        prev = s
        for suf in _EN_SUFFIXES:
            s = re.sub(rf"{suf}$", "", s)
    return s


def pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV/TSV with tolerant parsing for real-world lists (TCFD/SBTi).

    Strategy:
    1) Try common delimiters with the fast C engine.
    2) If that fails, try python engine with explicit delimiters.
    3) As a last resort, skip bad lines.

    This avoids pandas' delimiter sniffer error: "Could not determine delimiter".
    """

    # Common delimiters seen in public lists
    candidates = [",", "\t", ";", "|", " "]

    # 1) C engine, explicit delimiters
    for sep in candidates[:4]:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # 2) Python engine, explicit delimiters
    for sep in candidates:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # 3) Last resort: python engine + skip bad lines
    warnings.warn(f"Falling back to skipping bad lines for: {path}")
    for sep in candidates:
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                sep=sep,
                engine="python",
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    raise ValueError(f"Could not parse file with common delimiters: {path}")


def read_company_list_txt(path: Path) -> pd.DataFrame:
    """Read a 1-company-per-line text file and return DataFrame with company_name."""
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    companies = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # de-dup preserving order
    seen = set()
    uniq = []
    for c in companies:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    return pd.DataFrame({"company_name": uniq})


def resolve_sbti_path(path: Path) -> Path | None:
    """Resolve SBTi dashboard path robustly.

    Handles filename variants such as:
    - SBTi's Target Dashboard.csv
    - SBTis_Target_Dashboard.csv
    - SBTis Target Dashboard.csv
    - any *SBTi*.csv under the same directory
    """
    if path.exists():
        return path

    raw_dir = path.parent
    candidates = [
        raw_dir / "SBTi's Target Dashboard.csv",
        raw_dir / "SBTis_Target_Dashboard.csv",
        raw_dir / "SBTis Target Dashboard.csv",
    ]

    # Fallback: any csv containing 'SBTi'
    candidates.extend(sorted(raw_dir.glob("*SBTi*.csv")))

    for c in candidates:
        if c.exists():
            return c
    return None


# -----------------------------
# Core builder
# -----------------------------
def build_flags(
    project_root: Path,
    universe_gx_path: Path,
    jpx_map_path: Path,
    jpx_english_map_path: Path,
    tcfd_raw_path: Path,
    sbti_raw_path: Path,
    out_path: Path,
    out_unmatched_path: Path,
    opco_universe_path: Path,
    scores_out_path: Path,
) -> None:
    # Load required files
    gx = pd.read_csv(universe_gx_path, encoding="utf-8-sig")
    jpx = pd.read_csv(jpx_map_path, encoding="utf-8-sig")
    jpx_en = pd.DataFrame()
    if jpx_english_map_path.exists():
        jpx_en = pd.read_csv(jpx_english_map_path, encoding="utf-8-sig")
    else:
        print(
            f"[WARN] English name map not found: {jpx_english_map_path} "
            "- SBTi matching will fall back to 0 if no names are available."
        )

    print(f"[INFO] Loaded universe_gx: {universe_gx_path} shape={gx.shape}")
    print(f"[INFO] Loaded jpx_map: {jpx_map_path} shape={jpx.shape}")
    if not jpx_en.empty:
        print(f"[INFO] Loaded jpx English map: {jpx_english_map_path} shape={jpx_en.shape}")

    def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # strip BOM/whitespace from headers
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df

    gx = _canonicalize_columns(gx)
    jpx = _canonicalize_columns(jpx)
    if not jpx_en.empty:
        jpx_en = _canonicalize_columns(jpx_en)

    # Accept common alternatives and normalize to internal names
    gx_ticker_col = pick_first_existing_column(gx, ["securities_code", "ticker", "銘柄コード", "証券コード", "code"])
    if gx_ticker_col is None:
        raise ValueError(f"Missing ticker/code column in {universe_gx_path}. Columns: {list(gx.columns)}")

    gx_flag_col = pick_first_existing_column(gx, ["gx_flag", "is_gx", "gx", "gxleague_flag", "gx_league_flag"])
    if gx_flag_col is None:
        raise ValueError(f"Missing GX flag column in {universe_gx_path}. Columns: {list(gx.columns)}")

    jpx_ticker_col = pick_first_existing_column(jpx, ["securities_code", "ticker", "銘柄コード", "証券コード", "code"])
    if jpx_ticker_col is None:
        raise ValueError(f"Missing ticker/code column in {jpx_map_path}. Columns: {list(jpx.columns)}")

    # Rename to internal canonical names
    gx = gx.rename(columns={gx_ticker_col: "ticker", gx_flag_col: "is_gx"})
    jpx = jpx.rename(columns={jpx_ticker_col: "ticker"})

    if "company_name_norm" not in jpx.columns:
        # If missing, derive from company_name
        if "company_name" not in jpx.columns:
            raise ValueError(
                f"Missing 'company_name_norm' and 'company_name' in {jpx_map_path}. "
                "Need at least one to normalize."
            )
        jpx["company_name_norm"] = jpx["company_name"].map(normalize_company_name)

    # Standardize ticker formatting (keep leading zeros)
    gx["ticker"] = gx["ticker"].astype(str).str.strip().str.zfill(4)
    jpx["ticker"] = jpx["ticker"].astype(str).str.strip().str.zfill(4)
    if not jpx_en.empty and "ticker" in jpx_en.columns:
        jpx_en["ticker"] = jpx_en["ticker"].astype(str).str.strip().str.zfill(4)
        if "english_name_norm" in jpx_en.columns:
            jpx_en["english_name_norm"] = (
                jpx_en["english_name_norm"].fillna("").astype(str).map(normalize_english_name)
            )

    # Optional filter: operating companies only
    gx_size_before = len(gx)
    if opco_universe_path.exists():
        opco = pd.read_csv(opco_universe_path, encoding="utf-8-sig")
        opco_tickers = set(opco.get("ticker", pd.Series(dtype=str)).astype(str).str.strip().str.zfill(4))
        gx = gx[gx["ticker"].isin(opco_tickers)].copy()
        print(
            f"[INFO] Operating-company filter applied: {gx_size_before} -> {len(gx)} "
            f"({opco_universe_path})"
        )
    else:
        print(f"[WARN] Operating-company universe not found: {opco_universe_path} (no filter applied)")

    # Keep only necessary columns for join
    gx_base = gx[["ticker", "is_gx"]].copy()
    gx_base["is_gx"] = gx_base["is_gx"].fillna(0).astype(int)

    # --- TCFD ---
    # Prefer the cleaned 1-company-per-line list (tcfd_company_names.txt).
    # If a .csv is provided, fall back to robust CSV parsing.
    if tcfd_raw_path.suffix.lower() == ".txt":
        tcfd_raw = read_company_list_txt(tcfd_raw_path)
        tcfd_name_col = "company_name"
    else:
        tcfd_raw = read_csv_robust(tcfd_raw_path)
        print(f"[INFO] Loaded tcfd_raw: {tcfd_raw_path} shape={tcfd_raw.shape}")
        tcfd_name_col = pick_first_existing_column(
            tcfd_raw,
            [
                "company_name",
                "Company Name",
                "COMPANY_NAME",
                "企業名",
                "会社名",
                "提出会社名",
                "名称",
            ],
        )
        if tcfd_name_col is None:
            raise ValueError(
                f"Could not find a company name column in {tcfd_raw_path}. "
                f"Columns: {list(tcfd_raw.columns)}"
            )

    print(f"[INFO] Loaded tcfd list: {tcfd_raw_path} shape={tcfd_raw.shape}")
    tcfd = tcfd_raw.copy()
    tcfd["company_name_norm"] = tcfd[tcfd_name_col].map(normalize_company_name)
    tcfd = tcfd.merge(
        jpx[["ticker", "company_name_norm"]],
        on="company_name_norm",
        how="left",
        validate="m:1",
    )
    tcfd["tcfd_flag"] = 1
    tcfd_flags = (
        tcfd.dropna(subset=["ticker"])[["ticker", "tcfd_flag"]]
        .drop_duplicates(subset=["ticker"])
        .copy()
    )

    # --- SBTi ---
    # SBTi list is in English; JPX names are Japanese. Bridge via yfinance English names
    # (long/short) collected in jpx_ticker_english_name.csv to avoid relying on ISIN/LEI.
    resolved_sbti = resolve_sbti_path(sbti_raw_path)

    if resolved_sbti is None:
        print(f"[WARN] SBTi file not found: {sbti_raw_path} -> sbti_flag will be all 0")
        sbti_flags = pd.DataFrame({"ticker": gx_base["ticker"].unique(), "sbti_flag": 0})
    else:
        if resolved_sbti != sbti_raw_path:
            print(f"[INFO] Resolved SBTi path: {resolved_sbti}")
        sbti_raw = read_csv_robust(resolved_sbti)
        print(f"[INFO] Loaded sbti_raw: {resolved_sbti} shape={sbti_raw.shape}")

        sbti_name_col = pick_first_existing_column(
            sbti_raw,
            [
                "company_name",
                "Company Name",
                "COMPANY_NAME",
                "Company",
                "Organisation",
                "Organization",
                "企業名",
                "会社名",
            ],
        )
        if sbti_name_col is None:
            raise ValueError(
                f"Could not find a company name column in {resolved_sbti}. "
                f"Columns: {list(sbti_raw.columns)}"
            )

        sbti = sbti_raw.copy()
        sbti["english_name_norm"] = sbti[sbti_name_col].map(normalize_english_name)

        if jpx_en.empty:
            matched_cnt = 0
            total_cnt = len(sbti)
            print(
                "[WARN] SBTi matching skipped because English name map is missing. "
                f"Match rate: {matched_cnt}/{total_cnt} (0.0%)"
            )
            sbti["ticker"] = None
            sbti_flags = pd.DataFrame({"ticker": gx_base["ticker"].unique(), "sbti_flag": 0})
        else:
            # English-name bridge: normalize SBTi English names and join to yfinance English names
            if "english_name_norm" not in jpx_en.columns:
                # derive if missing to stay resilient
                jpx_en = jpx_en.copy()
                jpx_en["english_name_norm"] = jpx_en["yfinance_long_name"].fillna("").map(normalize_english_name)
                fallback_mask = jpx_en["english_name_norm"] == ""
                jpx_en.loc[fallback_mask, "english_name_norm"] = (
                    jpx_en.loc[fallback_mask, "yfinance_short_name"].fillna("").map(normalize_english_name)
                )

            sbti = sbti.merge(
                jpx_en[["ticker", "english_name_norm"]],
                on="english_name_norm",
                how="left",
                validate="m:1",
            )
            matched_cnt = sbti["ticker"].notna().sum()
            total_cnt = len(sbti)
            match_rate = matched_cnt / total_cnt if total_cnt else 0
            print(f"[INFO] SBTi English-name matched tickers: {matched_cnt}/{total_cnt} ({match_rate:.1%})")
            sbti["sbti_flag"] = 1
            sbti_flags = (
                sbti.dropna(subset=["ticker"])[["ticker", "sbti_flag"]]
                .drop_duplicates(subset=["ticker"])
                .copy()
            )

    # Merge all flags onto universe
    out = gx_base.merge(tcfd_flags, on="ticker", how="left")
    out = out.merge(sbti_flags, on="ticker", how="left")

    out["tcfd_flag"] = out["tcfd_flag"].fillna(0).astype(int)
    out["sbti_flag"] = out["sbti_flag"].fillna(0).astype(int)

    # Write main output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Write unmatched rows for manual QA (very useful)
    tcfd_unmatched = tcfd[tcfd["ticker"].isna()][[tcfd_name_col, "company_name_norm"]].copy()
    tcfd_unmatched["source"] = "TCFD"

    sbti_unmatched = pd.DataFrame()
    if resolved_sbti is not None:
        sbti_unmatched = sbti[sbti["ticker"].isna()][[sbti_name_col, "english_name_norm"]].copy()
        sbti_unmatched = sbti_unmatched.rename(columns={sbti_name_col: tcfd_name_col})
        sbti_unmatched["source"] = "SBTi"

    unmatched = pd.concat([tcfd_unmatched, sbti_unmatched], ignore_index=True)
    unmatched = unmatched.drop_duplicates()

    out_unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(out_unmatched_path, index=False)

    # Scores output
    scores = out.copy()
    scores["commitment_score"] = (
        scores["is_gx"] * _COMMITMENT_WEIGHT_GX
        + scores["tcfd_flag"] * _COMMITMENT_WEIGHT_TCFD
        + scores["sbti_flag"] * _COMMITMENT_WEIGHT_SBTI
    )
    scores_out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_out_path, index=False)

    # Quick stats
    n_before = gx_size_before
    n_after = len(out)
    tcfd_cov = out["tcfd_flag"].sum()
    sbti_cov = out["sbti_flag"].sum()
    gx_cov = out["is_gx"].sum()

    denom = n_after if n_after else 1
    print("[OK] climate_commitment_flags.csv generated")
    print(f"Universe size: before filter={n_before}, after filter={n_after}")
    print(f"GX coverage:   {gx_cov} ({gx_cov/denom:.1%})")
    print(f"TCFD coverage: {tcfd_cov} ({tcfd_cov/denom:.1%})")
    print(f"SBTi coverage: {sbti_cov} ({sbti_cov/denom:.1%})")
    print(f"Unmatched rows written to: {out_unmatched_path}")
    print(f"[OK] climate_commitment_scores.csv generated: {scores_out_path}")

    # Score summary + top 10
    desc = scores["commitment_score"].describe()
    print(
        f"[SUMMARY] commitment_score min={desc['min']:.0f} "
        f"p25={desc['25%']:.0f} median={desc['50%']:.0f} "
        f"p75={desc['75%']:.0f} max={desc['max']:.0f}"
    )
    jpx_names = jpx[["ticker", "company_name"]].drop_duplicates(subset=["ticker"]) if "company_name" in jpx.columns else None
    top10 = scores.sort_values(["commitment_score", "ticker"], ascending=[False, True]).head(10)
    if jpx_names is not None:
        top10 = top10.merge(jpx_names, on="ticker", how="left")
    display_cols = ["ticker", "commitment_score", "is_gx", "tcfd_flag", "sbti_flag"]
    if "company_name" in top10.columns:
        display_cols.insert(1, "company_name")
    print("[TOP 10] commitment_score")
    print(top10[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument(
        "--universe_gx",
        default="src/data_collection/processed/universe_with_gx_flag.csv",
        help="Path to universe_with_gx_flag.csv",
    )
    parser.add_argument(
        "--jpx_map",
        default="src/data_collection/processed/jpx_company_ticker_map.csv",
        help="Path to jpx_company_ticker_map.csv",
    )
    parser.add_argument(
        "--jpx_english_map",
        default="src/data_collection/processed/jpx_ticker_english_name.csv",
        help="Path to JPX English name map (from yfinance)",
    )
    parser.add_argument(
        "--tcfd_raw",
        default="src/data_collection/processed/tcfd_company_names.txt",
        help="Path to TCFD list (prefer tcfd_company_names.txt; can also accept CSV)",
    )
    parser.add_argument(
        "--sbti_raw",
        default="src/data_collection/raw/SBTis_Target_Dashboard.csv",
        help="Path to SBTi dashboard CSV (filename variants auto-detected; optional)",
    )
    parser.add_argument(
        "--out",
        default="src/data_collection/processed/climate_commitment_flags.csv",
        help="Output path",
    )
    parser.add_argument(
        "--out_unmatched",
        default="src/data_collection/processed/climate_commitment_flags_unmatched.csv",
        help="Unmatched output path",
    )
    parser.add_argument(
        "--opco_universe",
        default="src/data_collection/processed/universe_operating_companies.csv",
        help="Operating-company universe (optional filter if file exists)",
    )
    parser.add_argument(
        "--scores_out",
        default="src/data_collection/processed/climate_commitment_scores.csv",
        help="Output path for commitment scores",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    build_flags(
        project_root=root,
        universe_gx_path=root / args.universe_gx,
        jpx_map_path=root / args.jpx_map,
        jpx_english_map_path=root / args.jpx_english_map,
        tcfd_raw_path=root / args.tcfd_raw,
        sbti_raw_path=root / args.sbti_raw,
        out_path=root / args.out,
        out_unmatched_path=root / args.out_unmatched,
        opco_universe_path=root / args.opco_universe,
        scores_out_path=root / args.scores_out,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
