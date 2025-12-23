#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rank GX companies by Energy-origin CO2 emission trends (EEGS) and select Top-N.

Input:
  graph_long_from_raw.csv (output of fetch_env_make_array.py)

Output:
  (Output folder is created under the given 2nd screening directory)
  _analysis_emission_trend_YYYYMMDD_HHMMSS/
    - gx_company_emission_summary_energyCO2.csv
    - gx_top40_energyCO2_trend_based.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graph_long_csv",
        required=True,
        help="Path to graph_long_from_raw.csv"
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="2nd screening directory where analysis folder will be created"
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=40,
        help="Number of companies to select (default: 40)"
    )
    args = ap.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(args.graph_long_csv)

    # Safety check
    required_cols = {"company_name", "spEmitCode", "year", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort for YoY calculation
    df = df.sort_values(["spEmitCode", "year"]).copy()

    # -----------------------------
    # YoY calculation
    # -----------------------------
    df["yoy"] = df.groupby("spEmitCode")["value"].pct_change()

    # -----------------------------
    # Aggregate per company
    # -----------------------------
    agg = (
        df.groupby(["company_name", "spEmitCode"])
          .agg(
              n_years=("year", "nunique"),
              max_year=("year", "max"),
              mean_emission=("value", "mean"),
              mean_yoy=("yoy", "mean"),
              std_yoy=("yoy", "std"),
          )
          .reset_index()
    )

    # -----------------------------
    # Screening conditions
    # -----------------------------
    screened = agg[
        (agg["n_years"] >= 5) &
        (agg["max_year"] >= 2022) &
        (agg["mean_yoy"].notna())
    ].copy()

    # -----------------------------
    # Scoring
    # -----------------------------
    screened["impact"] = screened["mean_emission"] * screened["n_years"]
    screened["trend"] = - screened["mean_yoy"] / (1 + screened["std_yoy"].fillna(0))
    screened["final_score"] = screened["impact"] * screened["trend"]

    screened = screened.sort_values("final_score", ascending=False)

    top40 = screened.head(args.top_n).copy()

    # -----------------------------
    # Output directory
    # -----------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_root,
        f"_analysis_emission_trend_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Save outputs
    # -----------------------------
    summary_path = os.path.join(
        out_dir,
        "gx_company_emission_summary_energyCO2.csv"
    )
    top40_path = os.path.join(
        out_dir,
        "gx_top40_energyCO2_trend_based.csv"
    )

    screened.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top40.to_csv(top40_path, index=False, encoding="utf-8-sig")

    print("[OK] Output directory:", out_dir)
    print("[OK] Summary CSV:", summary_path, "rows=", len(screened))
    print("[OK] Top40 CSV:", top40_path, "rows=", len(top40))


if __name__ == "__main__":
    main()