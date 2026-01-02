#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_rank_energyCO2_impact_trend.py

Purpose:
  Build a robust cross-sectional ranking for Energy/CO2-related indicators
  using the *long-format* EEGS panel (eegs_panel.csv) as input.
  - Handles missing values (skip per-metric scoring)
  - Handles outliers via winsorization
  - Normalizes with percentile ranks (distribution-robust)
  - Produces metric scores + weighted composite + final rank
  - Saves both rank panel and summary (auditability)

Typical usage:
  python 10_rank_energyCO2_impact_trend.py \
    --input eegs_panel.csv \
    --output_dir /path/to/run/10_rank_energyco2 \
    --id_col spEmitCode \
    --year_col year \
    --gasid_col gasID \
    --value_col value \
    --unit_col unit \
    --repdiv_col repDivID \
    --year latest \
    --gas_dict_source /path/to/eegs_long.csv \
    --gas_dict_source /path/to/eegs_long_dir

  If --gas_dict_source is provided (can be repeated, file or directory), the script emits gasid_dictionary.csv (and gasid_dictionary_freq.csv) into output_dir.

Input requirements:
  - Must include id_col (e.g., spEmitCode), year_col, gasid_col, value_col
  - Input file must be a *long-format* EEGS panel (not wide)
  - Metric columns configured in METRICS will be derived from the panel

Outputs:
  - energyco2_rank_panel.csv
  - energyco2_rank_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration (EDIT HERE)
# -----------------------------

@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    weight: float = 1.0
    winsorize_p: float = 0.01          # 1% winsorization by default
    transform: Optional[str] = None    # None | "log1p" | "signed_log1p"
    required: bool = False             # if True and missing -> error


# This script is designed to work directly with the *long-format* eegs panel:
#   id_col, year_col, gasid_col, value_col, (optional) unit_col, (optional) repdiv_col
# It will pivot gasID -> columns and then derive wide metrics used for ranking.

# Map gasID -> semantic fields (confirmed via gasid_dictionary.csv).
# IMPORTANT:
#   - gasID=0 is "合計" (total). Do NOT add it together with its components (double-count risk).
#   - gasID=1 is "エネルギー起源CO2" (Energy-origin CO2).
#   - gasID=20 is "調整後排出量" (Adjusted emissions).
#   - gasID=9 is NOT revenue; it is "エネルギー起源CO2（発電所等配分前）".
GASID_MAP: Dict[str, List[int]] = {
    # Phase D primary axis
    "energy_co2": [1],
    # Robust headline figure
    "adjusted_emissions": [20],
    # Aggregate total (already includes components)
    "total_emissions": [0],
    # Optional components (coverage varies)
    "non_energy_co2": [2],
    "waste_fuel_non_energy_co2": [3],
    # Optional alternative energy CO2 definition (coverage limited)
    "energy_co2_pre_allocation": [9],
}

# Unit scale conversion for common Japanese unit strings.
# If unit is missing/unknown, values are left as-is.
UNIT_SCALE: Dict[str, float] = {
    "": 1.0,
    "千": 1_000.0,
    "百万": 1_000_000.0,
    "億": 100_000_000.0,
}

# Derived wide metrics computed from the long panel.
# These names are what METRICS refers to.
DERIVED_METRICS: List[str] = [
    "energy_co2",
    "adjusted_emissions",
    "total_emissions",
    "non_energy_co2",
    "energy_co2_pre_allocation",
]

# Ranking direction:
#   - Emissions levels: smaller is better => higher_is_better=False
METRICS: List[MetricSpec] = [
    # Phase D primary metric (required)
    MetricSpec(name="energy_co2", higher_is_better=False, weight=1.0, winsorize_p=0.01, transform="signed_log1p", required=True),
    # Secondary robustness check (may be missing for a subset)
    MetricSpec(name="adjusted_emissions", higher_is_better=False, weight=0.7, winsorize_p=0.01, transform="signed_log1p", required=False),
    # Total emissions (useful sanity check)
    MetricSpec(name="total_emissions", higher_is_better=False, weight=0.5, winsorize_p=0.01, transform="signed_log1p", required=False),
    # Optional component
    MetricSpec(name="non_energy_co2", higher_is_better=False, weight=0.3, winsorize_p=0.01, transform="signed_log1p", required=False),
    # Optional alternative definition
    MetricSpec(name="energy_co2_pre_allocation", higher_is_better=False, weight=0.2, winsorize_p=0.01, transform="signed_log1p", required=False),
]
# -----------------------------
# Utility functions
# -----------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def apply_unit_scale(values: pd.Series, units: Optional[pd.Series]) -> pd.Series:
    """Convert values to base units using UNIT_SCALE if unit strings exist."""
    v = _to_numeric(values)
    if units is None:
        return v
    u = units.fillna("").astype(str)
    scale = u.map(lambda x: UNIT_SCALE.get(x, 1.0))
    return v * _to_numeric(scale)


def pivot_long_panel(
    df_long: pd.DataFrame,
    id_col: str,
    year_col: str,
    gasid_col: str,
    value_col: str,
    unit_col: Optional[str] = None,
    repdiv_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert long panel (id, year, gasID, value, unit) -> wide (gasID_* columns).

    Notes:
      - If repdiv_col is provided and present, we aggregate (sum) across repDivID.
      - We scale values by unit if unit_col is provided.
      - For duplicated (id, year, gasID) rows, we sum.
    """
    required = [id_col, year_col, gasid_col, value_col]
    for c in required:
        if c not in df_long.columns:
            raise KeyError(f"Missing required column in long panel: {c}")

    units = df_long[unit_col] if (unit_col is not None and unit_col in df_long.columns) else None
    vals = apply_unit_scale(df_long[value_col], units)

    work = df_long.copy()
    work[value_col] = vals

    # Optional: aggregate across repDivID
    group_keys = [id_col, year_col, gasid_col]
    agg = (
        work[group_keys + [value_col]]
        .groupby(group_keys, as_index=False)[value_col]
        .sum()
    )

    wide = agg.pivot_table(
        index=[id_col, year_col],
        columns=gasid_col,
        values=value_col,
        aggfunc="sum",
    )

    # Flatten columns: gasID integers -> gas_<id>
    wide.columns = [f"gas_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()
    return wide


def sum_gasids(wide: pd.DataFrame, gasids: List[int]) -> pd.Series:
    if not gasids:
        return pd.Series(np.nan, index=wide.index)
    cols = [f"gas_{int(g)}" for g in gasids if f"gas_{int(g)}" in wide.columns]
    if not cols:
        return pd.Series(np.nan, index=wide.index)
    return wide[cols].sum(axis=1, min_count=1)


def build_derived_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    """Create derived metrics used in ranking from GASID_MAP."""
    out = wide.copy()

    # Levels (units are already scaled by apply_unit_scale in pivot_long_panel)
    out["energy_co2"] = sum_gasids(out, GASID_MAP.get("energy_co2", []))
    out["adjusted_emissions"] = sum_gasids(out, GASID_MAP.get("adjusted_emissions", []))
    out["total_emissions"] = sum_gasids(out, GASID_MAP.get("total_emissions", []))
    out["non_energy_co2"] = sum_gasids(out, GASID_MAP.get("non_energy_co2", []))
    out["energy_co2_pre_allocation"] = sum_gasids(out, GASID_MAP.get("energy_co2_pre_allocation", []))

    # NOTE:
    # Intensity metrics like *_per_revenue are intentionally NOT computed here because
    # the current gasID dictionary does not include revenue. If you later confirm a revenue gasID,
    # add it and introduce intensity metrics as a separate phase.

    return out


# -----------------------------
# GasID dictionary helper
# -----------------------------

def build_gasid_dictionary_from_sources(
    sources: List[str],
    max_files: int = 200,
) -> pd.DataFrame:
    """Build a gasID dictionary from eegs_long source CSVs.

    Each eegs_long CSV is expected to contain at least:
      - gasID
      - gas_name_raw
      - gas_name_norm
      - unit

    `sources` can include file paths and/or directory paths.
    The function returns a tidy table with counts so you can audit coverage.
    """
    file_paths: List[str] = []
    for s in sources:
        if os.path.isdir(s):
            file_paths.extend(sorted(glob.glob(os.path.join(s, "eegs_long_*.csv"))))
        else:
            file_paths.append(s)

    file_paths = [p for p in file_paths if os.path.exists(p)]
    if not file_paths:
        raise FileNotFoundError(f"No eegs_long source files found from: {sources}")

    file_paths = file_paths[:max_files]

    rows: List[pd.DataFrame] = []
    for p in file_paths:
        try:
            t = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read source file: {p} ({e})", file=sys.stderr)
            continue

        needed = ["gasID", "gas_name_raw", "gas_name_norm", "unit"]
        missing = [c for c in needed if c not in t.columns]
        if missing:
            print(f"[WARN] source file missing columns {missing}: {p}", file=sys.stderr)
            continue

        u = (
            t[["gasID", "gas_name_raw", "gas_name_norm", "unit"]]
            .drop_duplicates()
            .copy()
        )
        u["source_file"] = os.path.basename(p)
        rows.append(u)

    if not rows:
        raise RuntimeError("Could not build gasID dictionary: no usable eegs_long sources.")

    all_map = pd.concat(rows, ignore_index=True)

    # Summarize: for each gasID, list most frequent normalized name + unit
    def _mode(series: pd.Series) -> str:
        s = series.dropna().astype(str)
        if s.empty:
            return ""
        return s.value_counts().index[0]

    summary = (
        all_map
        .groupby("gasID", as_index=False)
        .agg(
            gas_name_norm=("gas_name_norm", _mode),
            gas_name_raw=("gas_name_raw", _mode),
            unit=("unit", _mode),
            n_variants_name_norm=("gas_name_norm", lambda x: int(x.dropna().nunique())),
            n_variants_unit=("unit", lambda x: int(x.dropna().nunique())),
            n_sources=("source_file", lambda x: int(x.nunique())),
        )
        .sort_values("gasID")
        .reset_index(drop=True)
    )

    # Add frequency table for debugging (optional but helpful)
    freq = (
        all_map
        .groupby(["gasID", "gas_name_norm", "unit"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["gasID", "count"], ascending=[True, False])
    )

    return summary, freq


# -----------------------------
# Utility functions
# -----------------------------

def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {ext}. Use .csv or .parquet")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def signed_log1p(x: pd.Series) -> pd.Series:
    # Allows log-like compression for both positive and negative values.
    # Useful when values can be near 0 but have skew.
    # sign(x) * log1p(abs(x))
    return np.sign(x) * np.log1p(np.abs(x))


def apply_transform(s: pd.Series, transform: Optional[str]) -> pd.Series:
    if transform is None:
        return s
    if transform == "log1p":
        # Only valid for >= 0; if negatives exist, they will become NaN
        return np.where(s >= 0, np.log1p(s), np.nan)
    if transform == "signed_log1p":
        return signed_log1p(s)
    raise ValueError(f"Unknown transform: {transform}")


def winsorize_series(s: pd.Series, p: float) -> Tuple[pd.Series, float, float]:
    """
    Winsorize by clamping to [p, 1-p] quantiles.
    Returns (winsorized_series, lo, hi)
    """
    if s.dropna().empty:
        return s, np.nan, np.nan
    lo = float(s.quantile(p))
    hi = float(s.quantile(1.0 - p))
    return s.clip(lower=lo, upper=hi), lo, hi


def percentile_score(s: pd.Series) -> pd.Series:
    """
    Robust normalization: percentile rank in [0,1].
    NaN stays NaN.
    """
    # rank(pct=True) assigns percent rank. Use method="average" for ties.
    return s.rank(pct=True, method="average")


def metric_to_score(df: pd.DataFrame, spec: MetricSpec) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Convert raw metric column to a score in [0,1] with:
      - optional transform
      - winsorize
      - direction alignment (higher is better)
      - percentile scoring
    """
    col = spec.name
    raw = df[col].astype("float64")

    # Transform
    transformed = pd.Series(apply_transform(raw, spec.transform), index=df.index, name=col)

    # Winsorize (on transformed)
    w, lo, hi = winsorize_series(transformed, spec.winsorize_p)

    # Direction: score should be higher-is-better
    aligned = w if spec.higher_is_better else (-w)

    # Percentile score
    score = percentile_score(aligned)

    audit = {
        "metric": col,
        "higher_is_better": spec.higher_is_better,
        "weight": spec.weight,
        "winsorize_p": spec.winsorize_p,
        "winsor_lo": lo,
        "winsor_hi": hi,
        "transform": spec.transform,
        "n_total": int(len(raw)),
        "n_nonnull_raw": int(raw.notna().sum()),
        "n_nonnull_used": int(pd.Series(aligned).notna().sum()),
        "missing_rate_raw": float(1.0 - raw.notna().mean()),
    }
    return score, audit


def choose_year(df: pd.DataFrame, year_col: str, year: str) -> pd.DataFrame:
    """
    year:
      - "latest": keep only max(year_col) per company
      - numeric string: keep that year
      - "all": keep all rows (then ranking must be adjusted; we keep it simple)
    """
    if year == "all":
        return df

    if year == "latest":
        # Keep latest year per id later (requires id_col); here just filter max year overall if no id_col grouping.
        # We will do per-company latest if id_col exists in main flow.
        return df

    # numeric year
    try:
        target = int(year)
    except ValueError:
        raise ValueError(f"--year must be one of: latest | all | <int year>. Got: {year}")

    return df[df[year_col] == target].copy()


# -----------------------------
# Main pipeline
# -----------------------------

def build_rank(
    df: pd.DataFrame,
    id_col: str,
    year_col: Optional[str],
    year: str,
    specs: List[MetricSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Basic checks
    if id_col not in df.columns:
        raise KeyError(f"Missing id_col '{id_col}' in input columns")

    # If year column provided, filter
    if year_col is not None:
        if year_col not in df.columns:
            raise KeyError(f"Missing year_col '{year_col}' in input columns")

        df = choose_year(df, year_col, year)

        if year == "latest":
            # choose latest per company_id
            df = (
                df.sort_values([id_col, year_col])
                  .groupby(id_col, as_index=False)
                  .tail(1)
                  .copy()
            )

    # Ensure unique entity rows for ranking
    # If duplicates remain, keep the last occurrence after sorting (deterministic)
    df = df.sort_values([id_col]).drop_duplicates(subset=[id_col], keep="last").copy()

    # Validate metric columns
    existing = set(df.columns)
    missing_required = [m.name for m in specs if m.required and m.name not in existing]
    if missing_required:
        raise KeyError(f"Missing required metric columns: {missing_required}")

    # Compute per-metric scores
    audits: List[Dict[str, object]] = []
    score_cols: List[str] = []

    for spec in specs:
        if spec.name not in df.columns:
            # skip optional missing
            print(f"[WARN] metric column not found, skipped: {spec.name}", file=sys.stderr)
            audits.append({
                "metric": spec.name,
                "skipped": True,
                "reason": "column_not_found",
                "weight": spec.weight,
            })
            continue

        score, audit = metric_to_score(df, spec)
        colname = f"score__{spec.name}"
        df[colname] = score
        score_cols.append(colname)
        audit["skipped"] = False
        audits.append(audit)

    if not score_cols:
        raise RuntimeError("No metric scores were computed. Check METRICS configuration and input columns.")

    # Weighted composite: normalize weights over available metrics (not skipped)
    available_specs = [s for s in specs if f"score__{s.name}" in df.columns]
    weights = np.array([s.weight for s in available_specs], dtype="float64")

    if np.all(weights == 0):
        raise ValueError("All available metric weights are zero. Adjust METRICS weights.")

    # Normalize weights to sum to 1 for interpretability
    weights = weights / weights.sum()

    # Composite score: mean of available (non-missing) weighted scores
    # We compute row-wise: sum(w_i * score_i) over non-missing; re-normalize by sum of weights actually present.
    score_matrix = df[[f"score__{s.name}" for s in available_specs]].to_numpy(dtype="float64")
    present = ~np.isnan(score_matrix)

    weighted = score_matrix * weights.reshape(1, -1)
    weighted_sum = np.nansum(weighted, axis=1)
    weight_present_sum = np.sum(present * weights.reshape(1, -1), axis=1)

    composite = np.where(weight_present_sum > 0, weighted_sum / weight_present_sum, np.nan)
    df["score__total"] = composite

    # Final rank: higher score is better; rank 1 is best
    df["rank__total"] = df["score__total"].rank(ascending=False, method="min")

    # Helpful diagnostics: how many metrics each company got scored on
    df["n_scored_metrics"] = np.sum(~df[score_cols].isna().to_numpy(), axis=1)

    # Output panel
    base_cols = [id_col]
    if year_col is not None and year_col in df.columns:
        base_cols.append(year_col)

    out_cols = base_cols + [c for c in df.columns if c.startswith("score__")] + ["rank__total", "n_scored_metrics"]
    rank_panel = df[out_cols].sort_values(["rank__total", id_col]).reset_index(drop=True)

    # Summary (auditable)
    summary_rows: List[Dict[str, object]] = []
    for a in audits:
        summary_rows.append(a)

    summary = pd.DataFrame(summary_rows)

    # Add composite info to summary
    summary = pd.concat(
        [
            summary,
            pd.DataFrame([{
                "metric": "__TOTAL__",
                "skipped": False,
                "method": "weighted_percentile_composite",
                "available_metrics": [s.name for s in available_specs],
                "normalized_weights": {s.name: float(w) for s, w in zip(available_specs, weights)},
                "n_entities": int(len(rank_panel)),
                "year_mode": year,
            }])
        ],
        ignore_index=True
    )

    return rank_panel, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV/Parquet file (LONG panel like eegs_panel.csv)")
    p.add_argument("--output_dir", required=True, help="Output directory for 10_rank_energyco2 results")

    # Long-format schema
    p.add_argument("--id_col", default="spEmitCode", help="Entity identifier column in long panel")
    p.add_argument("--year_col", default="year", help="Year column name in long panel")
    p.add_argument("--gasid_col", default="gasID", help="gasID column name in long panel")
    p.add_argument("--value_col", default="value", help="Value column name in long panel")
    p.add_argument("--unit_col", default="unit", help="Unit column name in long panel (optional)")
    p.add_argument("--repdiv_col", default="repDivID", help="repDivID column name in long panel (optional)")

    # Ranking scope
    p.add_argument("--year", default="latest", help="latest | all | <year int>")

    p.add_argument(
        "--gas_dict_source",
        action="append",
        default=[],
        help=(
            "Optional. Path to an eegs_long_*.csv file or a directory containing eegs_long_*.csv files. "
            "Can be passed multiple times. If provided, the script writes gasid_dictionary.csv and gasid_dictionary_freq.csv to output_dir."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df_long = read_table(args.input)
    ensure_dir(args.output_dir)

    if args.gas_dict_source:
        try:
            gasid_summary, gasid_freq = build_gasid_dictionary_from_sources(args.gas_dict_source)
            gasid_summary_path = os.path.join(args.output_dir, "gasid_dictionary.csv")
            gasid_freq_path = os.path.join(args.output_dir, "gasid_dictionary_freq.csv")
            gasid_summary.to_csv(gasid_summary_path, index=False)
            gasid_freq.to_csv(gasid_freq_path, index=False)
            print(f"[OK] Wrote gasID dictionary: {gasid_summary_path}")
            print(f"[OK] Wrote gasID dictionary freq: {gasid_freq_path}")
        except Exception as e:
            print(f"[WARN] Failed to build gasID dictionary from sources: {e}", file=sys.stderr)

    # Treat unit_col / repdiv_col as optional if they do not exist
    unit_col = args.unit_col if args.unit_col in df_long.columns else None
    repdiv_col = args.repdiv_col if args.repdiv_col in df_long.columns else None

    wide = pivot_long_panel(
        df_long=df_long,
        id_col=args.id_col,
        year_col=args.year_col,
        gasid_col=args.gasid_col,
        value_col=args.value_col,
        unit_col=unit_col,
        repdiv_col=repdiv_col,
    )

    wide = build_derived_metrics(wide)

    # Rank using derived metrics (wide)
    rank_panel, summary = build_rank(
        df=wide,
        id_col=args.id_col,
        year_col=args.year_col,
        year=args.year,
        specs=METRICS,
    )

    panel_path = os.path.join(args.output_dir, "energyco2_rank_panel.csv")
    summary_path = os.path.join(args.output_dir, "energyco2_rank_summary.csv")

    # Overwrite existing outputs in output_dir (intentional; makes reruns deterministic)
    rank_panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Wrote rank panel:   {panel_path}")
    print(f"[OK] Wrote rank summary: {summary_path}")
    print(f"[INFO] Entities ranked: {len(rank_panel)}")

    # Guidance for configuration
    if any(len(GASID_MAP.get(k, [])) == 0 for k in ["energy_total", "renewable_share"]):
        print(
            "[NOTE] GASID_MAP has empty mappings for some fields (e.g., energy_total / renewable_share). "
            "Those derived metrics will be NaN and automatically down-weighted per entity. "
            "If you know the correct gasIDs, set them in GASID_MAP at the top of this file.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())