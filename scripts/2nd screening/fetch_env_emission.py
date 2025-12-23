#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase B: Link screened companies to Japan Environment (EEGS) index.

Inputs
------
- screened_2_5.csv: 2.5次スクリーニング済み企業（252社想定）
- Japan_environment_2023.csv: 環境省EEGS側の索引（事業者名 + spEmitCode 等）

Outputs (timestamped folder)
----------------------------
- screened_2_5_with_envlink.csv
- matched_companies.csv
- unmatched_companies.csv
- env_index_cleaned.csv
- match_report.json
- config_used.json

Notes
-----
- This script does NOT fetch timeseries from EEGS graph yet.
- Its job is to link each screened company to spEmitCode (if possible) and
  to preserve unmatched/ambiguous cases for audit.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional fuzzy lib
try:
    from rapidfuzz import fuzz  # type: ignore

    def fuzzy_ratio(a: str, b: str) -> float:
        return float(fuzz.ratio(a, b))  # 0..100

    FUZZY_SCALE = 100.0
except Exception:
    import difflib

    def fuzzy_ratio(a: str, b: str) -> float:
        return float(difflib.SequenceMatcher(None, a, b).ratio())  # 0..1

    FUZZY_SCALE = 1.0


# -----------------------------
# Normalization
# -----------------------------
_CORP_TOKENS = [
    "株式会社",
    "（株）",
    "(株)",
    "有限会社",
    "合同会社",
    "ホールディングス",
    "ホールディング",
    "ホールディングス株式会社",
    "hd",
    "ＨＤ",
    "グループ",
    "co.,ltd",
    "co., ltd",
    "company",
    "limited",
    "inc",
    "incorporated",
]

_PUNCT_RE = re.compile(
    r"[\s\u3000\-‐-‒–—―_\.,，、・･\(\)（）\[\]{}<>＜＞/\\&＆'\"“”‘’·:：;；!！?？]+"
)


def normalize_company_name(name: Any) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""

    # width-ish normalization (minimal): unify spaces, lower
    s = s.replace("　", " ").strip().lower()

    # drop corp tokens
    for t in _CORP_TOKENS:
        s = s.replace(t, "")

    # drop punctuation/spaces
    s = _PUNCT_RE.sub("", s)

    return s


# -----------------------------
# Column detection
# -----------------------------

def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cset = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cset:
            return cset[cand.lower()]
    return None


def detect_company_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    cand = _pick_first_existing(
        cols,
        [
            "gx_company_name",
            "mapped_company_name",
            "rank_company_name",
            "company_jp",
            "company_name_jp",
            "company_name",
            "company",
            "name",
            "企業名",
            "銘柄名",
            "社名",
            "正規化社名",
            "norm_name",
            "normname",
        ],
    )
    if cand:
        return cand
    # fallback: first column containing '名'
    for c in cols:
        if "名" in c:
            return c
    raise ValueError(f"Could not detect company name column. Please pass --screen_name_col. Columns={cols}")


def detect_sp_emit_code_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    cand = _pick_first_existing(
        cols,
        [
            "spEmitCode",
            "sp_emit_code",
            "sp_emitcode",
            "排出事業者コード",
            "排出事業者ｺｰﾄﾞ",
            "事業者コード",
            "事業者ｺｰﾄﾞ",
        ],
    )
    if cand:
        return cand
    for c in cols:
        if "emit" in c.lower() or "コード" in c:
            return c
    raise ValueError(f"Could not detect spEmitCode column. Columns={cols}")


# -----------------------------
# Matching
# -----------------------------

@dataclass
class BestMatch:
    env_key: str
    env_raw_name: str
    sp_emit_codes: List[str]
    score: float


def build_env_index(env_df: pd.DataFrame, env_name_col: str, sp_col: str) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    df = env_df.copy()
    df["_env_name_raw"] = df[env_name_col].astype(str)
    df["_env_key"] = df["_env_name_raw"].map(normalize_company_name)
    df["_spEmitCode"] = df[sp_col].astype(str)

    df = df[df["_env_key"].astype(bool)].copy()

    grouped = (
        df.groupby("_env_key")
        .agg(
            env_names=("_env_name_raw", lambda x: sorted({str(v) for v in x if str(v).strip()})),
            sp_emit_codes=("_spEmitCode", lambda x: sorted({str(v) for v in x if str(v).strip()})),
            rows=("_spEmitCode", "size"),
        )
        .reset_index()
    )

    index: Dict[str, Dict[str, Any]] = {}
    for _, r in grouped.iterrows():
        index[str(r["_env_key"])] = {
            "env_names": list(r["env_names"]),
            "sp_emit_codes": list(r["sp_emit_codes"]),
            "rows": int(r["rows"]),
        }

    env_clean = grouped.rename(columns={"_env_key": "env_key"})
    env_clean["env_names"] = env_clean["env_names"].map(lambda xs: ";".join(xs))
    env_clean["sp_emit_codes"] = env_clean["sp_emit_codes"].map(lambda xs: ";".join(xs))

    return index, env_clean


def rule_candidates(screen_key: str, env_keys: List[str], min_len: int = 4) -> List[str]:
    if len(screen_key) < min_len:
        return []
    out = []
    for ek in env_keys:
        if len(ek) < min_len:
            continue
        if screen_key in ek or ek in screen_key:
            out.append(ek)
    return out


def fuzzy_candidates(screen_key: str, env_keys: List[str], topk: int) -> List[Tuple[str, float]]:
    scored = [(ek, fuzzy_ratio(screen_key, ek)) for ek in env_keys if ek]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def classify_match(
    screen_key: str,
    env_index: Dict[str, Dict[str, Any]],
    env_keys: List[str],
    threshold: float,
    topk: int,
) -> Tuple[str, Optional[BestMatch], List[Tuple[str, float]]]:
    # 1) EXACT
    if screen_key in env_index:
        rec = env_index[screen_key]
        best = BestMatch(
            env_key=screen_key,
            env_raw_name=(rec["env_names"][0] if rec["env_names"] else ""),
            sp_emit_codes=list(rec["sp_emit_codes"]),
            score=FUZZY_SCALE,
        )
        return "EXACT", best, [(screen_key, FUZZY_SCALE)]

    # 2) RULE
    rule = rule_candidates(screen_key, env_keys)
    if len(rule) == 1:
        ek = rule[0]
        rec = env_index[ek]
        best = BestMatch(
            env_key=ek,
            env_raw_name=(rec["env_names"][0] if rec["env_names"] else ""),
            sp_emit_codes=list(rec["sp_emit_codes"]),
            score=FUZZY_SCALE,
        )
        return "RULE", best, [(ek, FUZZY_SCALE)]

    if len(rule) > 1:
        scored = [(ek, fuzzy_ratio(screen_key, ek)) for ek in rule]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:topk]
        ek, sc = scored[0]
        rec = env_index[ek]
        best = BestMatch(
            env_key=ek,
            env_raw_name=(rec["env_names"][0] if rec["env_names"] else ""),
            sp_emit_codes=list(rec["sp_emit_codes"]),
            score=sc,
        )
        return "AMBIGUOUS", best, scored

    # 3) FUZZY
    scored = fuzzy_candidates(screen_key, env_keys, topk)
    if not scored:
        return "NONE", None, []

    best_ek, best_s = scored[0]
    if best_s >= threshold:
        # margin check to mark ambiguous when too close
        if len(scored) >= 2 and (best_s - scored[1][1]) < (0.03 * FUZZY_SCALE):
            rec = env_index[best_ek]
            best = BestMatch(
                env_key=best_ek,
                env_raw_name=(rec["env_names"][0] if rec["env_names"] else ""),
                sp_emit_codes=list(rec["sp_emit_codes"]),
                score=best_s,
            )
            return "AMBIGUOUS", best, scored

        rec = env_index[best_ek]
        best = BestMatch(
            env_key=best_ek,
            env_raw_name=(rec["env_names"][0] if rec["env_names"] else ""),
            sp_emit_codes=list(rec["sp_emit_codes"]),
            score=best_s,
        )
        return "FUZZY", best, scored

    return "NONE", None, scored


# -----------------------------
# Main
# -----------------------------

def make_outdir(out_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_root / f"_generated_envlink_{ts}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Link screened_2_5 to Japan_environment_2023 (Phase B)")
    ap.add_argument("--screened_csv", required=True, help="Path to screened_2_5.csv")
    ap.add_argument("--env_csv", required=True, help="Path to Japan_environment_2023.csv")
    ap.add_argument("--out_root", default=None, help="Output root dir (default: folder containing screened_csv)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=(90.0 if FUZZY_SCALE == 100.0 else 0.9),
        help="Fuzzy accept threshold",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top-K candidates to keep")
    ap.add_argument("--screen_name_col", default=None, help="Override column name for screened company name")
    ap.add_argument("--env_name_col", default=None, help="Override column name for env company name")
    ap.add_argument("--env_sp_col", default=None, help="Override column name for env spEmitCode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    screened_csv = Path(args.screened_csv)
    env_csv = Path(args.env_csv)

    out_root = Path(args.out_root) if args.out_root else screened_csv.parent
    outdir = make_outdir(out_root)
    outdir.mkdir(parents=True, exist_ok=True)

    screened_df = pd.read_csv(screened_csv)
    env_df = pd.read_csv(env_csv)

    screen_name_col = args.screen_name_col or detect_company_column(screened_df)
    env_name_col = args.env_name_col or detect_company_column(env_df)
    sp_col = args.env_sp_col or detect_sp_emit_code_column(env_df)

    # Validate override column names early
    for col, which in [(screen_name_col, "screen_name_col"), (env_name_col, "env_name_col"), (sp_col, "env_sp_col")]:
        if col not in (screened_df.columns if which == "screen_name_col" else env_df.columns):
            raise ValueError(f"Override {which}='{col}' not found in columns")

    env_index, env_clean = build_env_index(env_df, env_name_col, sp_col)
    env_keys = sorted(env_index.keys())

    screened_df = screened_df.copy()
    screened_df["_screen_name_raw"] = screened_df[screen_name_col].astype(str)
    screened_df["_screen_key"] = screened_df["_screen_name_raw"].map(normalize_company_name)

    stats = {"TOTAL": 0, "EXACT": 0, "RULE": 0, "FUZZY": 0, "AMBIGUOUS": 0, "NONE": 0}

    matched_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    ambiguous_examples: List[Dict[str, Any]] = []

    for idx, r in screened_df.iterrows():
        stats["TOTAL"] += 1
        raw = r["_screen_name_raw"]
        key = r["_screen_key"]

        if not key:
            stats["NONE"] += 1
            unmatched_rows.append(
                {
                    "row_index": int(idx),
                    "company_name": raw,
                    "normalized_name": key,
                    "reason": "EMPTY_NAME",
                    "top_candidates": "[]",
                }
            )
            continue

        status, best, cand_list = classify_match(key, env_index, env_keys, args.threshold, args.topk)
        stats[status] += 1

        cand_json = json.dumps(
            [{"env_key": ek, "score": float(sc)} for ek, sc in cand_list],
            ensure_ascii=False,
        )

        if status == "NONE" or best is None:
            unmatched_rows.append(
                {
                    "row_index": int(idx),
                    "company_name": raw,
                    "normalized_name": key,
                    "reason": "NO_MATCH",
                    "top_candidates": cand_json,
                }
            )
            continue

        matched_rows.append(
            {
                "row_index": int(idx),
                "company_name": raw,
                "normalized_name": key,
                "env_match_status": status,
                "env_matched_key": best.env_key,
                "env_matched_name_raw": best.env_raw_name,
                "env_spEmitCode": ";".join(best.sp_emit_codes),
                "env_match_score": float(best.score),
                "env_candidate_count": int(len(best.sp_emit_codes)),
                "top_candidates": cand_json,
            }
        )

        if status == "AMBIGUOUS" and len(ambiguous_examples) < 50:
            ambiguous_examples.append(
                {
                    "row_index": int(idx),
                    "company_name": raw,
                    "normalized_name": key,
                    "top_candidates": json.loads(cand_json),
                }
            )

    matched_df = pd.DataFrame(matched_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    screened_with_idx = screened_df.reset_index().rename(columns={"index": "row_index"})
    merged = screened_with_idx.merge(
        matched_df.drop(columns=["company_name", "normalized_name"], errors="ignore"),
        on="row_index",
        how="left",
    )

    # outputs
    out_screened = outdir / "screened_2_5_with_envlink.csv"
    out_matched = outdir / "matched_companies.csv"
    out_unmatched = outdir / "unmatched_companies.csv"
    out_envclean = outdir / "env_index_cleaned.csv"
    out_report = outdir / "match_report.json"
    out_config = outdir / "config_used.json"

    merged.to_csv(out_screened, index=False)
    matched_df.to_csv(out_matched, index=False)
    unmatched_df.to_csv(out_unmatched, index=False)
    env_clean.to_csv(out_envclean, index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"screened_csv": str(screened_csv), "env_csv": str(env_csv)},
        "outdir": str(outdir),
        "stats": stats,
        "threshold": args.threshold,
        "topk": args.topk,
        "fuzzy_scale": FUZZY_SCALE,
        "ambiguous_examples": ambiguous_examples,
    }

    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    config = {
        "timestamp": report["timestamp"],
        "columns": {
            "screen_company_col": screen_name_col,
            "env_company_col": env_name_col,
            "env_spEmitCode_col": sp_col,
        },
        "normalization": {
            "corp_tokens_removed": _CORP_TOKENS,
            "punct_regex": _PUNCT_RE.pattern,
        },
        "matching": {
            "order": ["EXACT", "RULE", "FUZZY"],
            "threshold": args.threshold,
            "topk": args.topk,
            "margin_rule": "AMBIGUOUS if (top1-top2) < 0.03*fuzzy_scale",
        },
        "python": {"rapidfuzz_available": (FUZZY_SCALE == 100.0)},
    }

    with out_config.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[OK] Phase B (env index linking) complete")
    print("[OK] outdir:", outdir)
    print("[OK] screened+envlink:", out_screened)
    print("[OK] matched:", out_matched, "rows=", len(matched_df))
    print("[OK] unmatched:", out_unmatched, "rows=", len(unmatched_df))
    print("[OK] report:", out_report)


if __name__ == "__main__":
    main()
