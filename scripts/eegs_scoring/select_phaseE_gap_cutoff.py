#!/usr/bin/env python3
"""
PhaseEランキングから第三次スクリーニング候補を自動選定する。

対応手法
- gap  : PhaseEスコアの隣接差分（最大ギャップ）を「不連続点」とみなして閾値選定
- jenks: Fisher–Jenks（Natural Breaks）により score__phaseE を自然分類し、
         上位クラスから累積して target_selected を目指す。
         目標到達時に境界クラスが大きい場合、境界クラス内で
         n_scored_metrics → quality_weight → score__total → score__phaseE の優先順で
         必要数のみを追加（ただし min_selected は下回らない）。

入力（デフォルト）
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p2_75/phaseE_rank_panel.csv

必須カラム
- spEmitCode
- score__phaseE
任意（jenksの境界クラス内タイブレークで使用）
- n_scored_metrics, quality_weight, score__total

出力
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/<run_id>/
  - phaseE_gap_rank_panel.csv : gap列 + jenks関連列を追加した全ランキング
  - phaseE_selected.csv       : 第三次候補
  - selection_summary.md      : 境界・統計・説明文（レポ貼り付け用）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


DEFAULT_INPUT = Path("/Users/shou/hobby/CPX/nikkei-stock/data/scores/p2_75/phaseE_rank_panel.csv")
DEFAULT_OUT_BASE = Path("/Users/shou/hobby/CPX/nikkei-stock/data/scores/p3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select Phase3 candidates from PhaseE ranking by 'gap' or 'jenks' method."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    p.add_argument(
        "--run_id",
        type=str,
        default="20260103_p3_start_phaseE_gap_from_p2_75",
        help="Output folder name under data/scores/p3/",
    )
    p.add_argument(
        "--method",
        type=str,
        default="gap",
        choices=["gap", "jenks"],
        help="Selection method: gap (largest discontinuity) or jenks (natural breaks). Default: gap.",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="If set, filter rows by this year (e.g., 2023). If omitted, use all rows.",
    )
    # gap-specific
    p.add_argument(
        "--top_fraction",
        type=float,
        default=0.30,
        help="(gap only) Search the largest gap only within the top fraction (0-1). Default: 0.30 (top 30%).",
    )
    # common constraints
    p.add_argument(
        "--min_selected",
        type=int,
        default=20,
        help="Hard floor: ensure at least this many rows are selected. Default: 20.",
    )
    p.add_argument(
        "--target_selected",
        type=int,
        default=40,
        help="Soft target: aim to select around this many rows (used mainly in jenks). Default: 40.",
    )
    # jenks-specific
    p.add_argument(
        "--k_min",
        type=int,
        default=3,
        help="(jenks only) Minimum number of classes. Default: 3.",
    )
    p.add_argument(
        "--k_max",
        type=int,
        default=7,
        help="(jenks only) Maximum number of classes. Default: 7.",
    )
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------

def robust_z(s: pd.Series) -> pd.Series:
    """Robust z-score using median and MAD. (Not used by default; kept for future extensions.)"""
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return (x - med) * 0.0
    return (x - med) / (1.4826 * mad)


def jenks_breaks(values: List[float], n_classes: int) -> List[float]:
    """
    Fisher–Jenks natural breaks (1D) using dynamic programming.

    Parameters
    ----------
    values : list[float]
        Must be sorted ascending for stable behavior.
    n_classes : int
        Number of classes (k). Must be >=2.

    Returns
    -------
    breaks : list[float]
        Length k+1. breaks[0]=min, breaks[-1]=max.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if not values:
        raise ValueError("values is empty")

    # Ensure ascending
    vals = sorted([float(v) for v in values])
    n = len(vals)

    # DP tables
    lower = [[0] * (n_classes + 1) for _ in range(n + 1)]
    var = [[float("inf")] * (n_classes + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        lower[i][1] = 1
        var[i][1] = 0.0

    # Precompute prefix sums for fast variance computation
    # variance of vals[m..i] can be computed from sums
    prefix = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for i in range(1, n + 1):
        v = vals[i - 1]
        prefix[i] = prefix[i - 1] + v
        prefix_sq[i] = prefix_sq[i - 1] + v * v

    def seg_variance(m: int, i: int) -> float:
        """Variance * count (i.e., SSE) for segment vals[m..i], 1-indexed inclusive."""
        # m, i are 1..n
        count = i - m + 1
        s = prefix[i] - prefix[m - 1]
        ss = prefix_sq[i] - prefix_sq[m - 1]
        mean = s / count
        # SSE = sum(x^2) - 2*mean*sum(x) + count*mean^2
        return ss - 2.0 * mean * s + count * mean * mean

    for k in range(2, n_classes + 1):
        # at least k items to form k classes
        for i in range(k, n + 1):
            best_m = -1
            best_var = float("inf")
            # try last class starting at m..i
            for m in range(k, i + 1):
                this = var[m - 1][k - 1] + seg_variance(m, i)
                if this < best_var:
                    best_var = this
                    best_m = m
            lower[i][k] = best_m
            var[i][k] = best_var

    # backtrack breaks
    breaks = [0.0] * (n_classes + 1)
    breaks[0] = vals[0]
    breaks[-1] = vals[-1]

    k = n_classes
    i = n
    while k > 1:
        m = lower[i][k]
        if m is None or m <= 1:
            breaks[k - 1] = vals[0]
            i = 1
        else:
            breaks[k - 1] = vals[m - 1]
            i = m - 1
        k -= 1

    # Ensure non-decreasing breaks
    for j in range(1, len(breaks)):
        if breaks[j] < breaks[j - 1]:
            breaks[j] = breaks[j - 1]
    return breaks


def gvf(values_asc: List[float], breaks: List[float]) -> float:
    """Goodness of Variance Fit for Jenks breaks. values_asc must be ascending."""
    if not values_asc:
        return 0.0
    vals = values_asc
    n = len(vals)
    mean_all = sum(vals) / n
    sdam = sum((x - mean_all) ** 2 for x in vals)  # total variance (sum squares)

    # Within-class variance (sum squares)
    sdcm = 0.0
    # breaks length = k+1
    b0 = breaks[0]
    k = len(breaks) - 1
    # assign by interval and accumulate
    # We'll just scan vals (ascending) and break into bins.
    idx = 0
    for c in range(k):
        lo = breaks[c]
        hi = breaks[c + 1]
        bucket = []
        while idx < n and (vals[idx] <= hi or c == k - 1):
            if vals[idx] < lo and c > 0:
                idx += 1
                continue
            bucket.append(vals[idx])
            idx += 1
            if idx >= n:
                break
        if bucket:
            m = sum(bucket) / len(bucket)
            sdcm += sum((x - m) ** 2 for x in bucket)

    if sdam == 0:
        return 1.0
    return (sdam - sdcm) / sdam


def assign_jenks_class(x: float, breaks: List[float]) -> int:
    """Assign x into Jenks class index 0..k-1 based on breaks (len=k+1)."""
    k = len(breaks) - 1
    # last bin inclusive
    for c in range(k):
        lo = breaks[c]
        hi = breaks[c + 1]
        if c == k - 1:
            if x >= lo and x <= hi:
                return c
        else:
            if x >= lo and x <= hi:
                return c
    # fallback (shouldn't happen)
    if x < breaks[0]:
        return 0
    return k - 1


def _available_sort_keys(df: pd.DataFrame) -> List[Tuple[str, bool]]:
    """Return sort keys (col, ascending) that exist in df, in desired priority."""
    candidates = [
        ("n_scored_metrics", False),
        ("quality_weight", False),
        ("score__total", False),
        ("score__phaseE", False),
    ]
    keys = []
    for col, asc in candidates:
        if col in df.columns:
            keys.append((col, asc))
    return keys


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    keys = _available_sort_keys(df)
    if not keys:
        return df
    by = [k[0] for k in keys]
    ascending = [k[1] for k in keys]
    return df.sort_values(by=by, ascending=ascending)


def main() -> None:
    args = parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = DEFAULT_OUT_BASE / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    required_cols = {"spEmitCode", "score__phaseE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.year is not None:
        if "year" not in df.columns:
            raise ValueError("--year was provided but 'year' column does not exist in input")
        df = df[df["year"] == args.year].copy()

    df = df.dropna(subset=["score__phaseE"]).copy()
    df["score__phaseE"] = pd.to_numeric(df["score__phaseE"], errors="coerce")
    df = df.dropna(subset=["score__phaseE"]).copy()

    # sort descending for ranking outputs
    df = df.sort_values("score__phaseE", ascending=False).reset_index(drop=True)

    n = len(df)
    if n < 3:
        raise ValueError(f"Not enough rows after filtering: {n}")

    # always compute gap column for diagnostics
    df["gap_to_next"] = df["score__phaseE"] - df["score__phaseE"].shift(-1)

    method = args.method
    min_selected = int(args.min_selected)
    target_selected = int(args.target_selected)

    # Jenks-related columns (filled later if used)
    df["jenks_class"] = pd.NA
    df["jenks_k"] = pd.NA
    df["jenks_gvf"] = pd.NA

    selected: pd.DataFrame
    boundary_class: Optional[int] = None

    if method == "gap":
        top_fraction = float(args.top_fraction)
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError("--top_fraction must be within (0, 1]")

        search_end = max(2, int(n * top_fraction))
        gaps = df.loc[: search_end - 1, ["gap_to_next"]].dropna()
        if gaps.empty:
            raise ValueError("No valid gaps found in the search range")

        cut_idx = int(gaps["gap_to_next"].idxmax())

        if cut_idx + 1 < min_selected:
            cut_idx = min_selected - 1

        threshold = float(df.loc[cut_idx, "score__phaseE"])
        selected = df[df["score__phaseE"] >= threshold].copy()

        selected_count = int(len(selected))
        max_gap_value = float(df.loc[cut_idx, "gap_to_next"]) if pd.notna(df.loc[cut_idx, "gap_to_next"]) else float("nan")

        print("\n[RESULT] Phase3 entry selection")
        print(f"- method: {method}")
        print(f"- input: {in_path}")
        if args.year is not None:
            print(f"- year_filter: {args.year}")
        print(f"- rows_used: {n}")
        print(f"- top_fraction_search: {top_fraction:.2f} (top {search_end} rows)")
        print(f"- cut_index (0-based): {cut_idx}")
        print(f"- max_gap_at_cut: {max_gap_value:.6f}")
        print(f"- threshold(score__phaseE): {threshold:.6f}")
        print(f"- min_selected: {min_selected}")
        print(f"- selected_count: {selected_count}")
        print(f"- out_dir: {out_dir}")

        rationale_text = (
            "PhaseEスコアを降順に並べ、隣接スコア差（gap）の最大点を『不連続点』として採用した。\n"
            "ただし下位側のノイズで巨大ギャップが出ることを避けるため、上位top_fractionの範囲内で最大ギャップを探索した。\n"
        )

        summary_lines = [
            "# PhaseE Selection Summary\n",
            f"- method: {method}\n",
            f"- input: {in_path}\n",
            f"- rows_used: {n}\n",
        ]
        if args.year is not None:
            summary_lines.append(f"- year_filter: {args.year}\n")
        summary_lines += [
            f"- top_fraction_search: {top_fraction:.2f} (top {search_end} rows)\n",
            f"- cut_index (0-based): {cut_idx}\n",
            f"- threshold(score__phaseE): {threshold:.6f}\n",
            f"- min_selected: {min_selected}\n",
            f"- selected_count: {selected_count}\n\n",
            "## Rationale\n",
            rationale_text,
        ]

    elif method == "jenks":
        k_min = int(args.k_min)
        k_max = int(args.k_max)
        if k_min < 2:
            raise ValueError("--k_min must be >= 2")
        if k_max < k_min:
            raise ValueError("--k_max must be >= k_min")

        vals_asc = sorted(df["score__phaseE"].astype(float).tolist())

        best_k = None
        best_gvf = -1.0
        best_breaks = None

        for k in range(k_min, k_max + 1):
            br = jenks_breaks(vals_asc, k)
            g = gvf(vals_asc, br)
            if (g > best_gvf) or (g == best_gvf and (best_k is None or k < best_k)):
                best_k = k
                best_gvf = g
                best_breaks = br

        assert best_k is not None and best_breaks is not None

        # assign class to each row
        df["jenks_class"] = df["score__phaseE"].apply(lambda x: assign_jenks_class(float(x), best_breaks))
        df["jenks_k"] = best_k
        df["jenks_gvf"] = best_gvf

        # include from top class downward until reaching target
        included_classes = []
        cumulative = 0
        for c in range(best_k - 1, -1, -1):
            included_classes.append(c)
            cumulative += int((df["jenks_class"] == c).sum())
            if cumulative >= target_selected:
                boundary_class = c
                break

        if boundary_class is None:
            boundary_class = 0

        # initial selection: all rows in included classes
        selected = df[df["jenks_class"].isin(included_classes)].copy()
        selected_count = int(len(selected))

        # enforce hard floor: if somehow below min, add more classes (should rarely happen)
        if selected_count < min_selected:
            for c in range(boundary_class - 1, -1, -1):
                if c in included_classes:
                    continue
                included_classes.append(c)
                selected = df[df["jenks_class"].isin(included_classes)].copy()
                selected_count = int(len(selected))
                boundary_class = c
                if selected_count >= min_selected:
                    break

        # If overshoot: partial take from boundary class to get closer to target (but never below min)
        # Keep all higher classes fully; slice boundary class by tie-break keys.
        if selected_count > target_selected and boundary_class is not None:
            higher_classes = [c for c in included_classes if c > boundary_class]
            df_high = df[df["jenks_class"].isin(higher_classes)].copy()
            df_boundary = df[df["jenks_class"] == boundary_class].copy()

            high_count = int(len(df_high))
            need = target_selected - high_count

            # If need is too small, ensure min_selected
            if need < (min_selected - high_count):
                need = min_selected - high_count

            if need < 0:
                # even higher classes exceed target; in that case, just keep higher classes (but must satisfy min)
                # This is rare; fallback to keep top min_selected by phaseE.
                tmp = df_high.copy()
                tmp = _sort_df(tmp)
                selected = tmp.head(max(min_selected, target_selected)).copy()
                boundary_class = None
            else:
                df_boundary = _sort_df(df_boundary)
                df_boundary_take = df_boundary.head(need).copy()
                selected = pd.concat([df_high, df_boundary_take], axis=0).copy()

        selected = _sort_df(selected).reset_index(drop=True)
        selected_count = int(len(selected))

        print("\n[RESULT] Phase3 entry selection")
        print(f"- method: {method}")
        print(f"- input: {in_path}")
        if args.year is not None:
            print(f"- year_filter: {args.year}")
        print(f"- rows_used: {n}")
        print(f"- k_range: {k_min}..{k_max}")
        print(f"- chosen_k: {best_k}")
        print(f"- chosen_gvf: {best_gvf:.6f}")
        print(f"- target_selected: {target_selected}")
        print(f"- min_selected: {min_selected}")
        print(f"- boundary_class: {boundary_class}")
        print(f"- selected_count: {selected_count}")
        print(f"- out_dir: {out_dir}")

        rationale_text = (
            "PhaseEスコア（score__phaseE）に対してFisher–Jenks自然分類（Natural Breaks）を適用し、"
            "スコア分布の自然なまとまり（クラス）を抽出した。\n"
            "上位クラスから順に第三次候補に含め、目標社数（target_selected）に到達するまで累積した。\n"
            "目標到達時に境界クラスが大きい場合は、境界クラス内で開示の信頼性指標"
            "（n_scored_metrics, quality_weight, score__total）を優先して必要数のみを追加し、"
            "恣意性を抑えつつ候補数を調整した。\n"
        )

        summary_lines = [
            "# PhaseE Selection Summary\n",
            f"- method: {method}\n",
            f"- input: {in_path}\n",
            f"- rows_used: {n}\n",
        ]
        if args.year is not None:
            summary_lines.append(f"- year_filter: {args.year}\n")
        summary_lines += [
            f"- k_range: {k_min}..{k_max}\n",
            f"- chosen_k: {best_k}\n",
            f"- chosen_gvf: {best_gvf:.6f}\n",
            f"- target_selected: {target_selected}\n",
            f"- min_selected: {min_selected}\n",
            f"- boundary_class: {boundary_class}\n",
            f"- selected_count: {selected_count}\n\n",
            "## Rationale\n",
            rationale_text,
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Output
    panel_path = out_dir / "phaseE_gap_rank_panel.csv"
    selected_path = out_dir / "phaseE_selected.csv"
    summary_path = out_dir / "selection_summary.md"

    df.to_csv(panel_path, index=False)
    selected.to_csv(selected_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line)

        f.write("\n## Outputs\n")
        f.write(f"- {panel_path}\n")
        f.write(f"- {selected_path}\n")
        f.write(f"- {summary_path}\n")

    print("\n[OK] wrote outputs:")
    print(f"- {panel_path}")
    print(f"- {selected_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()