

#!/usr/bin/env python3
"""ranktop_enrich.py

Purpose
-------
Take an existing Phase3 rank_top output (phase3_ranktop_full.csv) and enrich it
with manually curated Notion-based qualitative evaluations (fromnotion.csv).

Design notes
------------
- This script does *not* change the Phase3 tagging/rules pipeline.
- It simply LEFT-JOINs extra columns by `entity_id` and writes an enriched CSV.
- The Notion export is currently modeled as generic columns (col1..col7) plus notes,
  because the Notion page did not include stable headers in the copied text.
  You can later rename/normalize these columns once you finalize the schema.

Usage
-----
python3 scripts/phase3/tagging/ranktop_enrich.py \
  --ranktop_full_csv /path/to/phase3_ranktop_full.csv \
  --fromnotion_csv /path/to/fromnotion.csv \
  --out_csv /path/to/phase3_ranktop_full_enriched.csv

Optional
--------
--fail_on_unmatched : if set, the script exits non-zero when no rows match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich Phase3 rank_top output with Notion-based qualitative CSV")
    p.add_argument("--ranktop_full_csv", type=Path, required=True, help="Input: phase3_ranktop_full.csv")
    p.add_argument("--fromnotion_csv", type=Path, required=True, help="Input: Notion manual export CSV")
    p.add_argument("--out_csv", type=Path, required=True, help="Output: enriched CSV")
    p.add_argument(
        "--fail_on_unmatched",
        action="store_true",
        help="Fail (exit 2) if the join produces zero matched rows (sanity check).",
    )
    return p.parse_args()


def _coerce_entity_id_to_str(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out


def main() -> int:
    args = parse_args()

    if not args.ranktop_full_csv.exists():
        raise FileNotFoundError(f"ranktop_full_csv not found: {args.ranktop_full_csv}")
    if not args.fromnotion_csv.exists():
        raise FileNotFoundError(f"fromnotion_csv not found: {args.fromnotion_csv}")

    rank_df = pd.read_csv(args.ranktop_full_csv)
    notion_df = pd.read_csv(args.fromnotion_csv)

    rank_df = _coerce_entity_id_to_str(rank_df, "entity_id")
    notion_df = _coerce_entity_id_to_str(notion_df, "entity_id")

    # Avoid accidental duplicate keys in notion_df (keep last as the most recent correction)
    if notion_df["entity_id"].duplicated().any():
        notion_df = notion_df.drop_duplicates(subset=["entity_id"], keep="last")

    merged = rank_df.merge(notion_df, on="entity_id", how="left", validate="many_to_one")

    matched = merged["col1"].notna().sum() if "col1" in merged.columns else 0
    total = len(merged)

    print(f"[INFO] Rows: {total} | Matched Notion rows: {matched}")

    if args.fail_on_unmatched and matched == 0:
        print("[ERROR] No rows matched between ranktop_full and fromnotion. Check entity_id alignment.")
        return 2

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[OK] enriched -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())