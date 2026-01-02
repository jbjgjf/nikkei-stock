#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""30_quality_weight_missingness.py

Phase E: Quality-weighted extension of Phase D ranking.

Purpose
-------
Convert Phase D "technical strength" into "evaluative / review strength"
by incorporating data quality signals:
- disclosure coverage
- gasID breadth
- missingness patterns (especially recent years)

Key principle
-------------
Quality weighting must NOT reward weak performers.
It only down-weights fragile or unreliable disclosures.

Formula
-------
  score__phaseE = score__total * quality_weight
  where quality_weight ∈ [0.5, 1.0]

Inputs
------
1) Phase D rank panel (from 10_rank_energyCO2_impact_trend.py)
2) EEGS long-format panel (eegs_panel.csv)

Outputs (written to output_dir)
-------------------------------
- phaseE_quality_features.csv
- phaseE_rank_panel.csv
- phaseE_summary.csv

Notes
-----
- This script is designed to be safe to re-run: it overwrites outputs.
- Defaults assume columns: spEmitCode, year, gasID.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# -----------------------------
# Configuration (EDITABLE)
# -----------------------------

QUALITY_WEIGHT_BOUNDS = (0.5, 1.0)

# Relative importance of quality components
W_COVERAGE = 0.6
W_BREADTH = 0.4
W_RECENT_MISSING_PENALTY = 0.2   # subtractive


# -----------------------------
# Helpers
# -----------------------------

def clamp(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# -----------------------------
# Quality feature construction
# -----------------------------

def compute_quality_features(
    df_long: pd.DataFrame,
    id_col: str,
    year_col: str,
    gasid_col: str,
) -> pd.DataFrame:
    """Compute per-company disclosure-quality indicators.

    Returns
    -------
    pd.DataFrame
        One row per company.
    """

    for c in (id_col, year_col, gasid_col):
        if c not in df_long.columns:
            raise KeyError(f"Missing column in long panel: {c}")

    work = df_long[[id_col, year_col, gasid_col]].copy()

    # Drop rows with missing essential identifiers
    work = work.dropna(subset=[id_col, year_col, gasid_col])

    # Ensure year is numeric
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=[year_col])
    work[year_col] = work[year_col].astype(int)

    # Reporting span per company
    span = (
        work.groupby(id_col)[year_col]
        .agg(min_year="min", max_year="max", n_years="nunique")
        .reset_index()
    )

    # gasID breadth
    gas_breadth = (
        work.groupby(id_col)[gasid_col]
        .nunique()
        .reset_index(name="n_gas_series")
    )

    # Observed gas-years (company-year-gas tuples)
    observed = (
        work.groupby(id_col)
        .size()
        .reset_index(name="observed_gas_years")
    )

    q = span.merge(gas_breadth, on=id_col, how="left").merge(observed, on=id_col, how="left")

    q["expected_gas_years"] = q["n_years"] * q["n_gas_series"]
    q["coverage_ratio"] = np.where(
        q["expected_gas_years"] > 0,
        q["observed_gas_years"] / q["expected_gas_years"],
        np.nan,
    )

    # Recent missingness: check last 2 years relative to each company's max_year
    recent_obs = work.merge(q[[id_col, "max_year", "n_gas_series"]], on=id_col, how="left")
    recent_obs = recent_obs[recent_obs[year_col] >= (recent_obs["max_year"] - 1)]

    recent_count = recent_obs.groupby(id_col).size().reset_index(name="recent_obs_count")

    q = q.merge(recent_count, on=id_col, how="left")
    q["recent_obs_count"] = q["recent_obs_count"].fillna(0)

    # Expected observations in last 2 years = 2 * n_gas_series
    q["expected_recent"] = 2 * q["n_gas_series"]
    q["missing_recent_flag"] = (q["recent_obs_count"] < q["expected_recent"]).astype(int)

    # Normalize breadth to [0,1]
    max_breadth = float(q["n_gas_series"].max()) if len(q) else 0.0
    q["n_gas_series_norm"] = (q["n_gas_series"] / max_breadth) if max_breadth > 0 else 0.0

    return q


# -----------------------------
# Quality weight construction
# -----------------------------

def build_quality_weight(q: pd.DataFrame) -> pd.DataFrame:
    """Construct bounded quality weights in [0.5, 1.0]."""

    raw = (
        W_COVERAGE * q["coverage_ratio"].fillna(0)
        + W_BREADTH * q["n_gas_series_norm"].fillna(0)
        - W_RECENT_MISSING_PENALTY * q["missing_recent_flag"].fillna(0)
    )

    q = q.copy()
    q["quality_weight_raw"] = raw
    q["quality_weight"] = clamp(raw, QUALITY_WEIGHT_BOUNDS[0], QUALITY_WEIGHT_BOUNDS[1])
    return q


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E: quality-weighted scoring")
    p.add_argument("--panel_phaseD", required=True, help="Phase D rank panel CSV (must include score__total)")
    p.add_argument("--eegs_long", required=True, help="EEGS long-format panel CSV")
    p.add_argument("--output_dir", required=True, help="Directory to write outputs")
    p.add_argument("--id_col", default="spEmitCode")
    p.add_argument("--year_col", default="year")
    p.add_argument("--gasid_col", default="gasID")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dfD = read_csv(args.panel_phaseD)
    dfL = read_csv(args.eegs_long)

    if "score__total" not in dfD.columns:
        raise KeyError("Phase D panel missing required column: score__total")

    # Quality features and weights
    q = compute_quality_features(dfL, args.id_col, args.year_col, args.gasid_col)
    q = build_quality_weight(q)

    # Merge weights into Phase D panel
    out = dfD.merge(q[[args.id_col, "quality_weight"]], on=args.id_col, how="left")
    out["quality_weight"] = out["quality_weight"].fillna(QUALITY_WEIGHT_BOUNDS[0])

    # Phase E score and rank
    out["score__phaseE"] = out["score__total"] * out["quality_weight"]
    out["rank__phaseE"] = out["score__phaseE"].rank(ascending=False, method="min")

    # Write outputs (overwrite)
    q_path = os.path.join(args.output_dir, "phaseE_quality_features.csv")
    panel_path = os.path.join(args.output_dir, "phaseE_rank_panel.csv")
    summary_path = os.path.join(args.output_dir, "phaseE_summary.csv")

    q.to_csv(q_path, index=False)
    out.sort_values("rank__phaseE").to_csv(panel_path, index=False)

    summary = pd.DataFrame([
        {
            "n_entities": int(len(out)),
            "quality_weight_min": float(out["quality_weight"].min()),
            "quality_weight_max": float(out["quality_weight"].max()),
            "coverage_weight": W_COVERAGE,
            "breadth_weight": W_BREADTH,
            "recent_missing_penalty": W_RECENT_MISSING_PENALTY,
            "weight_lower_bound": QUALITY_WEIGHT_BOUNDS[0],
            "weight_upper_bound": QUALITY_WEIGHT_BOUNDS[1],
        }
    ])
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Phase E quality features: {q_path}")
    print(f"[OK] Phase E rank panel:      {panel_path}")
    print(f"[OK] Phase E summary:         {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())