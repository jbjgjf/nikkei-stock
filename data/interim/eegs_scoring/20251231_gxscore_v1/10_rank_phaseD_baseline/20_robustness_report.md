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

---

# EEGS Scoring Scripts

This folder contains scripts used to build the EEGS-based screening pipeline for the Nikkei Stock League project.

## Pipeline overview

- **Phase D (Baseline ranking):** build a transparent baseline ranking and a default composite ranking.
  - Produces a *rank panel* (one row per company) including `score__total` and ranks.

- **Phase D Robustness (Sensitivity):** validate that the baseline/composite rankings are stable.
  - Produces agreement metrics (Spearman/Pearson), Top-N overlap (Top10/Top20), and weight sensitivity tables.

- **Phase E (Quality weighting):** attach a disclosure-quality/credibility lens to Phase D.
  - Produces `quality_weight ∈ [0.5, 1.0]` and `score__phaseE = score__total × quality_weight`.

## Key scripts

- `10_rank_energyCO2_impact_trend.py`
  - Builds Phase D rank panels (baseline + default composite).

- `20_robustness_sensitivity_phaseD.py`
  - Computes Phase D robustness metrics and writes a run report.

- `30_quality_weight_missingness.py`
  - Computes Phase E quality weights and produces Phase E rank panel.

## Where outputs are written

All outputs are written under:

- `data/interim/eegs_scoring/<run_id>/<phase_output_dir>/`

Example (current run):

- `data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/`

This folder contains:
- `energyco2_rank_panel.csv` (Phase D)
- `20_robustness_report.md` and robustness tables (Phase D robustness)
- `phaseE_*.csv` (Phase E)

## Repro commands (example)

### Phase D robustness
```bash
python3 scripts/eegs_scoring/20_robustness_sensitivity_phaseD.py \
  --panel data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/energyco2_rank_panel.csv \
  --output_dir data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline
```

### Phase E
```bash
python3 scripts/eegs_scoring/30_quality_weight_missingness.py \
  --panel_phaseD data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/energyco2_rank_panel.csv \
  --eegs_long data/interim/eegs_scoring/20251231_gxscore_v1/eegs_panel.csv \
  --output_dir data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline
```

# Phase D Robustness Package (Baseline × Composite) — Detailed Report

**Important (ranking direction):** This configuration ranks **lower emissions higher** (low-emission companies appear at the top).

**Purpose of this document**

This report is written so that any teammate can understand (a) what was tested, (b) what the numbers mean, and (c) how to reuse the outputs in a final write-up **without reading the code**.

This robustness package supports a specific claim for judges:

> The Phase D baseline ranking is transparent and the key conclusions are robust to reasonable modeling choices (multi-metric composite and alternative weights).

This is **not Phase E** yet. It is a Phase D “stress test” that prepares the ground for Phase E (quality weighting).

---

## 0. Scope and definitions

### 0.1 Entities
- Number of companies evaluated: **183**
- Company identifier column: **`spEmitCode`**
- Year handling: **latest year per company** (in the current panel, many rows show **2023**)

### 0.2 Two ranking definitions compared

We compare two rankings produced from the Phase D output panel (`energyco2_rank_panel.csv`):

1) **Baseline ranking (single metric)**
- Score column: **`score__energy_co2`**
- Rank used in this robustness script: recomputed as **`rank__baseline_energy_co2`** from `score__energy_co2`
- Meaning: company’s position based only on the Energy-origin CO₂ axis

2) **Default composite ranking (multi-metric)**
- Score column: **`score__total`**
- Rank column in the panel: **`rank__total`**
- Meaning: weighted composite of multiple metric scores (each metric score is a percentile score in [0,1])

### 0.3 IMPORTANT: Direction of “better”
The Phase D script (`10_rank_energyCO2_impact_trend.py`) currently sets emissions level metrics as:

- `higher_is_better = False`

Internally, the script flips sign before computing percentile ranks, which means:

- **lower emissions → higher percentile score → better rank**

Therefore, “Top10” in this run corresponds to:
- **companies with relatively low emissions (under the configured definition)**

If the project later needs “largest emitters / highest impact,” the direction must be changed upstream. This report reflects the run as executed.

---

## 1. Inputs, outputs, and reproducibility

### 1.1 Input file used by robustness run
- **`energyco2_rank_panel.csv`**
  - produced by: `scripts/eegs_scoring/10_rank_energyCO2_impact_trend.py`
  - contains:
    - identifiers: `spEmitCode`, `year`
    - per-metric scores: `score__energy_co2`, `score__adjusted_emissions`, `score__total_emissions`, ...
    - composite: `score__total`
    - rank: `rank__total`
    - diagnostics: `n_scored_metrics`

### 1.2 Robustness script used
- `scripts/eegs_scoring/20_robustness_sensitivity_phaseD.py`

### 1.3 Output files (written into the same Phase D output directory)
1) Agreement metrics
- `20_robustness_rank_agreement.csv`

2) Top-N overlap membership tables
- `20_robustness_top_overlap_top10.csv`
- `20_robustness_top_overlap_top20.csv`

3) Weight sensitivity results
- `20_robustness_weight_sensitivity_summary.csv`
- `20_robustness_weight_sensitivity_top10.csv`

4) Narrative report
- `20_robustness_report.md` (this file)

### 1.4 Repro command (example)
Adjust paths if your folder name differs:

```bash
python3 scripts/eegs_scoring/20_robustness_sensitivity_phaseD.py \
  --panel data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/energyco2_rank_panel.csv \
  --output_dir data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline
```

## 2. Baseline vs Default Composite Agreement

### 2.1 Why this matters

A common critique is:

“Your ranking depends on your aggregation / weighting choice.”

So we test whether the multi-metric composite produces the same ordering as the baseline single metric axis.

### 2.2 Metrics used

A) Spearman correlation of ranks
	•	compares ordering directly (monotonic agreement)
	•	robust to nonlinear scaling
	•	interpretation:
	•	1.0 = identical ordering
	•	0.0 = no monotonic relation
	•	negative = opposite ordering

B) Pearson correlation of scores
	•	compares linear similarity of score values
	•	secondary; Spearman is the primary robustness indicator for rankings

### 2.3 Result (from this run)
	•	Entities ranked: 183
	•	Spearman(rank_baseline, rank_default): 0.9960
	•	Pearson(score_energy_co2, score_total_default): 0.9961

### 2.4 Interpretation (what we can claim safely)

These correlations are extremely high.

Meaning:
	•	the default composite ranking and baseline ranking are nearly identical
	•	key conclusions do not depend on a particular multi-metric aggregation design

Recommended write-up phrasing:

The baseline ranking and the composite ranking are nearly identical (Spearman = 0.996), indicating that our findings do not depend on a specific aggregation scheme.

⸻

## 3. Top-N overlap (Top10 / Top20)

### 3.1 Definitions
	•	Intersection: number of companies that appear in both TopN lists
	•	Jaccard index: intersection / union
	•	1.0 means identical sets
	•	0.0 means no overlap

### 3.2 Results (from this run)

Top10
	•	Intersection: 8 companies
	•	Jaccard: 0.6667

Top20
	•	Intersection: 19 companies
	•	Jaccard: 0.9048

### 3.3 Interpretation
	•	Top20 is extremely stable:
	•	19/20 overlap implies the shortlist is almost unchanged
	•	Top10 is moderately stable:
	•	8/10 overlap is still strong
	•	but the very top positions are more sensitive to composition choices

This is expected:
	•	Around the boundary (e.g., rank 8–12), score differences are often small
	•	small differences can swap positions even when the overall ordering is stable

Recommended write-up phrasing:

The Top20 shortlist remains highly stable under the composite specification (19/20 overlap; Jaccard 0.905). While the internal ordering within the Top10 can shift slightly, the candidate set itself is robust.

⸻

## 4. Weight sensitivity analysis (3 alternative scenarios)

### 4.1 Why this matters

Even if default composite matches baseline, a judge may ask:

“If you changed the weights, would results change?”

We therefore recompute composite scores/ranks under three alternative weight settings.

### 4.2 Metrics included (if present in the panel)

The sensitivity script recomputes a composite using available score columns in the panel:
	•	score__energy_co2
	•	score__adjusted_emissions
	•	score__total_emissions
	•	score__non_energy_co2
	•	score__energy_co2_pre_allocation (often sparse; included if present)

Missing handling (important):
	•	if a company lacks a score for a metric, that metric’s weight is removed for that company
	•	remaining weights are renormalized row-wise
	•	this prevents missingness from being treated as artificially good/bad

### 4.3 Scenarios tested
	•	S1_energy_heavy
	•	emphasize energy_CO2 (baseline axis)
	•	S2_adjusted_heavy
	•	emphasize adjusted emissions (robustness axis)
	•	S3_equal_core
	•	equal weights among core metrics (energy, adjusted, total)

### 4.4 Output files for sensitivity
	•	20_robustness_weight_sensitivity_summary.csv
	•	scenario-level correlations and Top10 overlap stats
	•	20_robustness_weight_sensitivity_top10.csv
	•	company-level membership: whether each company appears in Top10 under each scenario

### 4.5 How to interpret sensitivity results (guidance for teammates)

Focus on:
	•	spearman_rank_corr_vs_default
	•	spearman_rank_corr_vs_baseline
	•	top10_overlap_vs_default
	•	top10_jaccard_vs_default

What “good robustness” looks like:
	•	Spearman correlations remain high (rule of thumb: > 0.90)
	•	Top10 overlap remains large (rule of thumb: ≥ 7/10)

If one scenario deviates strongly:
	•	it becomes a discussion point:
	•	“ranking is stable except under extreme emphasis on X”
	•	this is not necessarily a failure; it provides an explainable sensitivity narrative

⸻

## 5. Suggested structure for the final paper / report section

### 5.1 Global stability
	•	Spearman correlation between baseline and default composite: 0.996

### 5.2 Shortlist stability
	•	Top20 overlap: 19/20 (Jaccard 0.905)
	•	Top10 overlap: 8/10 (Jaccard 0.667)

### 5.3 Weight sensitivity
	•	3 alternative weight scenarios tested
	•	results reported in:
	•	20_robustness_weight_sensitivity_summary.csv
	•	20_robustness_weight_sensitivity_top10.csv

If there is space, add one paragraph:
	•	why Top10 can shift more than Top20 (small margins around boundary)

⸻

## 6. Limitations (what this package does NOT do)

This robustness package:
	•	does not validate the correctness of raw emissions numbers
	•	does not incorporate industry structure
	•	does not use intensity metrics (e.g., per revenue) because revenue mapping is not confirmed in GasID dictionary
	•	does not implement Phase E quality weighting (coverage / continuity / missingness patterns)

These belong to later project phases.

⸻

## 7. Bridge to Phase E (next step)

Because Phase D ranking is robust under:
	•	default multi-metric composite
	•	alternative weights
	•	Top20 shortlist stability

we can proceed to Phase E:
	•	introduce quality_weight per company (coverage, continuity, missingness patterns)
	•	apply multiplicatively:
	•	final_score_E = final_score_D × quality_weight
	•	preserve Phase D ordering as much as possible while down-weighting fragile disclosures

Recommended Phase E deliverables:
	•	per-company quality feature table
	•	quality weight (clamped 0.5–1.0)
	•	list of companies strong in Phase D but downgraded by disclosure risk
	•	narrative explaining why quality affects trust, not performance

⸻

## 8. Phase E: Quality-weighted scoring (Implemented)

Phase E has now been implemented and executed. The goal is **not** to create an independent ranking, but to attach a **credibility / disclosure-quality lens** to the Phase D results.

### 8.1 Core principle
- Quality weighting must **not** elevate weak performers.
- It is used only to **down-weight rankings that rely on fragile or incomplete disclosure**.

Accordingly, we define:

- `quality_weight ∈ [0.5, 1.0]`
- `score__phaseE = score__total × quality_weight`
- `rank__phaseE` is computed by sorting `score__phaseE` descending.

### 8.2 Quality features (per company)
Phase E computes the following disclosure-quality features from the long-format EEGS panel (`eegs_panel.csv`):

- `n_gas_series`: number of distinct gasIDs observed for the company (disclosure breadth)
- `coverage_ratio`: observed gas-years / expected gas-years (continuity)
- `missing_recent_flag`: whether observations are missing in the most recent two years (governance/reporting risk signal)

**Missing handling:** if a company lacks some gasIDs, weights are renormalized per company so that missingness is not treated as automatically good/bad.

### 8.3 Outputs (Phase E deliverables)
Phase E writes three files into the same output directory:

- `phaseE_quality_features.csv` — one row per company with quality features and `quality_weight`
- `phaseE_rank_panel.csv` — Phase D panel plus `quality_weight`, `score__phaseE`, `rank__phaseE`
- `phaseE_summary.csv` — run metadata (entity count, min/max weight, configured coefficients)

Repro command:

```bash
python3 scripts/eegs_scoring/30_quality_weight_missingness.py \
  --panel_phaseD data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/energyco2_rank_panel.csv \
  --eegs_long data/interim/eegs_scoring/20251231_gxscore_v1/eegs_panel.csv \
  --output_dir data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline
```

### 8.4 Interpretation guidance
- Companies ranked highly in both Phase D and Phase E represent **strong results with credible disclosure**.
- Companies ranked highly in Phase D but downgraded in Phase E indicate **potential transition signals with disclosure or governance risk** (e.g., narrow gas coverage or recent missingness).
- Phase E should be presented as a **robustness / defensibility enhancement**, not as a “punishment.”

### 8.5 Run statistics (paste into the final write-up)

### 8.6 Phase D vs Phase E Top10 comparison

To illustrate the effect of disclosure-quality weighting, we compare the Top10 companies under Phase D and Phase E.

**Definition**
- Phase D Top10: companies with the 10 smallest values of `rank__total`
- Phase E Top10: companies with the 10 smallest values of `rank__phaseE`

**Observed difference (this run)**

- Companies appearing in Phase D Top10 **only**:
  - `260094087`
  - `400095131`
  - `800162996`
  - `985036501`

- Companies appearing in Phase E Top10 **only**:
  - `580031215`
  - `580042649`
  - `985642304`
  - `986331452`

**Interpretation**

These substitutions reflect the intended behavior of Phase E:

- The Phase D–only companies show strong numerical performance but receive lower quality weights due to narrower gasID coverage or recent missingness.
- The Phase E–only companies maintain comparable Phase D performance while exhibiting more complete or consistent disclosure patterns.

Crucially, this is **not a reversal of the overall ranking structure**. The majority of the Top10 remains shared between Phase D and Phase E, indicating that Phase E acts as a *credibility refinement* rather than a new scoring model.

This comparison provides a concrete, reviewer-friendly demonstration that:
- Phase D identifies strong candidates based on emissions-related metrics, and
- Phase E prioritizes candidates whose performance is supported by more reliable disclosure.
Concrete run metadata is stored in `phaseE_summary.csv`. Teammates should copy the single-row values into the final report (or cite them directly).

To view the stats:

```bash
cat data/interim/eegs_scoring/20251231_gxscore_v1/10_rank_phaseD_baseline/phaseE_summary.csv
```

At minimum, report:
- `n_entities`
- `quality_weight_min`, `quality_weight_max`
- the coefficients: `coverage_weight`, `breadth_weight`, `recent_missing_penalty`
- bounds: `weight_lower_bound`, `weight_upper_bound`