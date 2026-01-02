#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_rank_multigas_structure.py

目的（Phase Dの審査用パッケージ化：1,2,3）
(1) Baseline vs Composite の一致度（順位相関など）
(2) Top-N（10/20）の重なり表（overlap, Jaccard）
(3) 重み感度分析（3シナリオ）で結論の頑健性を示す

入力：
- 10_rankの出力: energyco2_rank_panel.csv

出力（すべて output_dir に保存）：
- 20_robustness_rank_agreement.csv
- 20_robustness_top_overlap_top10.csv
- 20_robustness_top_overlap_top20.csv
- 20_robustness_weight_sensitivity_summary.csv
- 20_robustness_weight_sensitivity_top10.csv
- 20_robustness_report.md

使い方：
python3 scripts/eegs_scoring/20_rank_multigas_structure.py \
  --panel /path/to/10_rank_energyco2/energyco2_rank_panel.csv \
  --output_dir /path/to/10_rank_energyco2
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_TOPNS = [10, 20]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(float)
    bb = b.astype(float)
    ok = aa.notna() & bb.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(aa[ok].corr(bb[ok], method="spearman"))


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(float)
    bb = b.astype(float)
    ok = aa.notna() & bb.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(aa[ok].corr(bb[ok], method="pearson"))


def compute_rank_from_score(score: pd.Series, ascending: bool = False) -> pd.Series:
    """rank 1 = best. ascending=False means higher score is better."""
    return score.rank(ascending=ascending, method="min")


def topn_ids(df: pd.DataFrame, id_col: str, rank_col: str, n: int) -> List[str]:
    t = df.sort_values(rank_col, ascending=True).head(n)
    return [str(x) for x in t[id_col].tolist()]


def overlap_stats(a: List[str], b: List[str]) -> Dict[str, float]:
    sa = set(a)
    sb = set(b)
    inter = sa & sb
    union = sa | sb
    return {
        "n_a": float(len(sa)),
        "n_b": float(len(sb)),
        "n_intersection": float(len(inter)),
        "jaccard": float(len(inter) / len(union)) if union else float("nan"),
        "overlap_rate_vs_a": float(len(inter) / len(sa)) if sa else float("nan"),
        "overlap_rate_vs_b": float(len(inter) / len(sb)) if sb else float("nan"),
    }


def build_top_overlap_table(
    df: pd.DataFrame,
    id_col: str,
    rank_a: str,
    rank_b: str,
    n: int,
    label_a: str,
    label_b: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    a = topn_ids(df, id_col, rank_a, n)
    b = topn_ids(df, id_col, rank_b, n)

    sa, sb = set(a), set(b)
    inter = sa & sb

    rows = []
    for i in sorted(sa | sb):
        rows.append(
            {
                id_col: i,
                f"in_{label_a}_top{n}": i in sa,
                f"in_{label_b}_top{n}": i in sb,
                "in_intersection": i in inter,
            }
        )

    stats = overlap_stats(a, b)
    return pd.DataFrame(rows), stats


def recompute_composite(
    df: pd.DataFrame,
    score_cols: List[str],
    weights: Dict[str, float],
    out_score_col: str,
    out_rank_col: str,
) -> pd.DataFrame:
    """10_rankと同じ：欠損を考慮して行ごとに重みを再正規化して合成。"""
    work = df.copy()

    w = np.array([float(weights.get(c, 0.0)) for c in score_cols], dtype="float64")
    if np.all(w == 0):
        raise ValueError("All weights are zero for the provided score_cols.")
    w = w / w.sum()

    mat = work[score_cols].to_numpy(dtype="float64")
    present = ~np.isnan(mat)

    weighted = mat * w.reshape(1, -1)
    weighted_sum = np.nansum(weighted, axis=1)
    w_present_sum = np.sum(present * w.reshape(1, -1), axis=1)

    comp = np.where(w_present_sum > 0, weighted_sum / w_present_sum, np.nan)
    work[out_score_col] = comp
    work[out_rank_col] = compute_rank_from_score(work[out_score_col], ascending=False)
    return work


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="Path to energyco2_rank_panel.csv")
    p.add_argument("--output_dir", required=True, help="Write outputs here (same 10_rank folder)")
    p.add_argument("--id_col", default="spEmitCode")
    p.add_argument("--year_col", default="year")
    p.add_argument("--topn", default="10,20", help="Comma-separated list: e.g., 10,20")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.panel)

    id_col = args.id_col
    if id_col not in df.columns:
        raise KeyError(f"Missing id_col '{id_col}'")

    baseline_score_col = "score__energy_co2"
    default_comp_col = "score__total"

    if baseline_score_col not in df.columns:
        raise KeyError(f"Missing column: {baseline_score_col}")
    if default_comp_col not in df.columns:
        raise KeyError(f"Missing column: {default_comp_col}")

    work = df.copy()
    work["rank__baseline_energy_co2"] = compute_rank_from_score(work[baseline_score_col], ascending=False)
    work["rank__default_composite"] = compute_rank_from_score(work[default_comp_col], ascending=False)

    # (1) Agreement
    agreement = {
        "n_entities": int(len(work)),
        "spearman_rank_corr": spearman_corr(work["rank__baseline_energy_co2"], work["rank__default_composite"]),
        "pearson_score_corr": pearson_corr(work[baseline_score_col], work[default_comp_col]),
    }
    agreement_path = os.path.join(args.output_dir, "20_robustness_rank_agreement.csv")
    pd.DataFrame([agreement]).to_csv(agreement_path, index=False)

    # (2) Top-N overlap
    topns = [int(x.strip()) for x in str(args.topn).split(",") if x.strip()]
    overlap_stats_rows = []
    for n in topns:
        tab, st = build_top_overlap_table(
            df=work,
            id_col=id_col,
            rank_a="rank__baseline_energy_co2",
            rank_b="rank__default_composite",
            n=n,
            label_a="baseline",
            label_b="default",
        )
        out_path = os.path.join(args.output_dir, f"20_robustness_top_overlap_top{n}.csv")
        tab.to_csv(out_path, index=False)
        overlap_stats_rows.append({"topn": n, **st})

    overlap_stats_df = pd.DataFrame(overlap_stats_rows).sort_values("topn").reset_index(drop=True)

    # (3) Weight sensitivity (3 scenarios)
    candidate_metrics = [
        "score__energy_co2",
        "score__adjusted_emissions",
        "score__total_emissions",
        "score__non_energy_co2",
        "score__energy_co2_pre_allocation",
    ]
    score_cols = [c for c in candidate_metrics if c in work.columns]

    scenarios: Dict[str, Dict[str, float]] = {
        "S1_energy_heavy": {
            "score__energy_co2": 0.60,
            "score__adjusted_emissions": 0.20,
            "score__total_emissions": 0.15,
            "score__non_energy_co2": 0.05,
        },
        "S2_adjusted_heavy": {
            "score__energy_co2": 0.35,
            "score__adjusted_emissions": 0.45,
            "score__total_emissions": 0.15,
            "score__non_energy_co2": 0.05,
        },
        "S3_equal_core": {
            "score__energy_co2": 1.0,
            "score__adjusted_emissions": 1.0,
            "score__total_emissions": 1.0,
        },
    }

    baseline_top10 = topn_ids(work, id_col, "rank__baseline_energy_co2", 10)
    default_top10 = topn_ids(work, id_col, "rank__default_composite", 10)

    sensitivity_rows = []
    top10_membership_rows = []

    for scen_name, wmap in scenarios.items():
        scen_score_col = f"score__{scen_name}"
        scen_rank_col = f"rank__{scen_name}"

        tmp = recompute_composite(
            df=work,
            score_cols=score_cols,
            weights=wmap,
            out_score_col=scen_score_col,
            out_rank_col=scen_rank_col,
        )

        scen_top10 = topn_ids(tmp, id_col, scen_rank_col, 10)

        ov_b = overlap_stats(baseline_top10, scen_top10)
        ov_d = overlap_stats(default_top10, scen_top10)

        sensitivity_rows.append(
            {
                "scenario": scen_name,
                "n_entities": int(len(tmp)),
                "spearman_rank_corr_vs_baseline": spearman_corr(tmp["rank__baseline_energy_co2"], tmp[scen_rank_col]),
                "spearman_rank_corr_vs_default": spearman_corr(tmp["rank__default_composite"], tmp[scen_rank_col]),
                "pearson_score_corr_vs_default": pearson_corr(tmp[default_comp_col], tmp[scen_score_col]),
                "top10_overlap_vs_baseline": ov_b["n_intersection"],
                "top10_jaccard_vs_baseline": ov_b["jaccard"],
                "top10_overlap_vs_default": ov_d["n_intersection"],
                "top10_jaccard_vs_default": ov_d["jaccard"],
            }
        )

        for x in sorted(set(baseline_top10) | set(default_top10) | set(scen_top10)):
            top10_membership_rows.append(
                {
                    "scenario": scen_name,
                    id_col: x,
                    "in_baseline_top10": x in set(baseline_top10),
                    "in_default_top10": x in set(default_top10),
                    "in_scenario_top10": x in set(scen_top10),
                }
            )

    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values("scenario").reset_index(drop=True)
    sens_path = os.path.join(args.output_dir, "20_robustness_weight_sensitivity_summary.csv")
    sensitivity_df.to_csv(sens_path, index=False)

    top10_df = pd.DataFrame(top10_membership_rows)
    top10_path = os.path.join(args.output_dir, "20_robustness_weight_sensitivity_top10.csv")
    top10_df.to_csv(top10_path, index=False)

    # Markdown report
    report_path = os.path.join(args.output_dir, "20_robustness_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Robustness Package (Phase D → Phase E)\n\n")
        f.write("This report summarizes the requested robustness checks (1)(2)(3).\n\n")

        f.write("## (1) Baseline vs Default Composite Agreement\n")
        f.write(f"- Entities ranked: **{agreement['n_entities']}**\n")
        f.write(f"- Spearman(rank_baseline, rank_default): **{agreement['spearman_rank_corr']:.4f}**\n")
        f.write(f"- Pearson(score_energy_co2, score_total_default): **{agreement['pearson_score_corr']:.4f}**\n\n")

        f.write("## (2) Top-N Overlap (Baseline vs Default Composite)\n")
        for _, r in overlap_stats_df.iterrows():
            n = int(r["topn"])
            f.write(f"### Top{n}\n")
            f.write(f"- Intersection: **{int(r['n_intersection'])}** companies\n")
            f.write(f"- Jaccard: **{float(r['jaccard']):.4f}**\n\n")

        f.write("## (3) Weight Sensitivity (3 Scenarios)\n")
        f.write("- S1_energy_heavy: emphasize energy_co2\n")
        f.write("- S2_adjusted_heavy: emphasize adjusted_emissions\n")
        f.write("- S3_equal_core: equal weights among core metrics\n\n")
        f.write("See `20_robustness_weight_sensitivity_summary.csv` for full numeric results.\n\n")

        f.write("## Output Files\n")
        f.write("- 20_robustness_rank_agreement.csv\n")
        for n in topns:
            f.write(f"- 20_robustness_top_overlap_top{n}.csv\n")
        f.write("- 20_robustness_weight_sensitivity_summary.csv\n")
        f.write("- 20_robustness_weight_sensitivity_top10.csv\n")
        f.write("- 20_robustness_report.md\n")

    print(f"[OK] Wrote: {agreement_path}")
    for n in topns:
        print(f"[OK] Wrote: {os.path.join(args.output_dir, f'20_robustness_top_overlap_top{n}.csv')}")
    print(f"[OK] Wrote: {sens_path}")
    print(f"[OK] Wrote: {top10_path}")
    print(f"[OK] Wrote: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())