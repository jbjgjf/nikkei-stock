#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_5 第2.5次スクリーニング
- financials_wide.csv を読み込み
- ハード除外ルールで落とす
- 残りをスコアリングしてランキング出力
出力:
  - screened_2_5.csv   : 残った企業 + スコア + 理由
  - excluded_2_5.csv   : 除外企業 + 除外理由
  - screening_2_5_summary.json : 件数サマリ
"""

import argparse
import json
import numpy as np
import pandas as pd


# -----------------------------
# 設定（必要ならここだけ調整）
# -----------------------------
HARD_THRESHOLDS = {
    "max_debt_to_equity": 3.0,
}

REQUIRED_FOR_HARD = [
    "market_cap",
    "revenue",
    "total_equity",
    "debt_to_equity",
    "op_margin",
    "net_margin",
]

# スコア重み（合計1.0）
WEIGHTS = {
    "profitability": 0.30,
    "growth": 0.20,
    "balance": 0.20,
    "cashflow": 0.15,
    "valuation": 0.15,
}

# 欠損ペナルティ（欠損1項目あたり）
MISSING_PENALTY_PER_ITEM = 2.0

# 外れ値クリップ（分位）
CLIP_Q_LOW = 0.01
CLIP_Q_HIGH = 0.99


def clip_series(s: pd.Series, q_low=CLIP_Q_LOW, q_high=CLIP_Q_HIGH) -> pd.Series:
    """外れ値を分位でクリップ"""
    if s.dropna().empty:
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def percentile_score(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    パーセンタイル順位を 0-100 に変換
    higher_is_better=True : 高いほど高得点
    False                : 低いほど高得点
    """
    # rank(pct=True) は欠損を保持
    pct = s.rank(pct=True, method="average")
    if not higher_is_better:
        pct = 1.0 - pct
    return (pct * 100.0).astype(float)


def hard_exclusion(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """ハード除外を適用し、(remain, excluded) を返す"""
    reasons = []

    # 必須欠損
    miss_mask = df[REQUIRED_FOR_HARD].isna().any(axis=1)
    reasons.append((miss_mask, "missing_required_fields"))

    # 成立性
    reasons.append((df["total_equity"].fillna(0) <= 0, "non_positive_equity"))
    reasons.append((df["market_cap"].fillna(0) <= 0, "non_positive_market_cap"))
    reasons.append((df["revenue"].fillna(0) <= 0, "non_positive_revenue"))

    # 収益性最低ライン（本業も最終も赤字）
    reasons.append((
        (df["operating_income"].fillna(0) < 0) & (df["net_income"].fillna(0) < 0),
        "both_operating_and_net_income_negative",
    ))

    # レバレッジ過多
    reasons.append((
        (df["debt_to_equity"].fillna(np.inf) > HARD_THRESHOLDS["max_debt_to_equity"]),
        f"debt_to_equity_gt_{HARD_THRESHOLDS['max_debt_to_equity']}",
    ))

    # CFが両方マイナス
    reasons.append((
        ((df["free_cf"].fillna(0) < 0) & (df["operating_cf"].fillna(0) < 0)),
        "both_free_cf_and_operating_cf_negative",
    ))

    exclude_reason = pd.Series([""] * len(df), index=df.index, dtype="object")
    excluded_any = pd.Series(False, index=df.index)

    for mask, label in reasons:
        mask = mask.fillna(False)
        excluded_any |= mask
        exclude_reason.loc[mask] = exclude_reason.loc[mask].where(
            exclude_reason.loc[mask] != "",
            label
        )
        # 既に理由が入っている場合は追記
        need_append = mask & (exclude_reason != label) & (exclude_reason != "")
        exclude_reason.loc[need_append] = exclude_reason.loc[need_append] + ";" + label

    excluded = df.loc[excluded_any].copy()
    excluded["exclude_reason"] = exclude_reason.loc[excluded_any].values

    remain = df.loc[~excluded_any].copy()
    remain["exclude_reason"] = ""  # 残るものは空

    return remain, excluded


def scoring(df: pd.DataFrame) -> pd.DataFrame:
    """スコア計算（0-100）"""
    out = df.copy()

    # 派生指標
    out["equity_ratio"] = out["total_equity"] / out["total_assets"]
    out["free_cf_margin"] = out["free_cf"] / out["revenue"]
    out["operating_cf_margin"] = out["operating_cf"] / out["revenue"]

    # クリップ対象
    clip_cols = [
        "op_margin", "net_margin", "roe", "roa",
        "revenue_yoy", "net_income_yoy",
        "equity_ratio", "free_cf_margin", "operating_cf_margin",
        "debt_to_equity", "pe", "pb", "ps_est",
    ]
    for c in clip_cols:
        if c in out.columns:
            out[c] = clip_series(out[c])

    # 個別スコア（欠損は後で50に置換）
    # 収益性
    out["s_op_margin"] = percentile_score(out["op_margin"], True)
    out["s_net_margin"] = percentile_score(out["net_margin"], True)
    out["s_roe"] = percentile_score(out["roe"], True)

    # 成長性
    out["s_revenue_yoy"] = percentile_score(out["revenue_yoy"], True)
    out["s_net_income_yoy"] = percentile_score(out["net_income_yoy"], True)

    # 健全性
    out["s_debt_to_equity"] = percentile_score(out["debt_to_equity"], False)
    out["s_equity_ratio"] = percentile_score(out["equity_ratio"], True)
    out["s_roa"] = percentile_score(out["roa"], True)

    # キャッシュ創出
    out["s_free_cf_margin"] = percentile_score(out["free_cf_margin"], True)
    out["s_operating_cf_margin"] = percentile_score(out["operating_cf_margin"], True)

    # バリュエーション（低いほど良い）
    out["s_pe"] = percentile_score(out["pe"], False)
    out["s_pb"] = percentile_score(out["pb"], False)
    out["s_ps"] = percentile_score(out["ps_est"], False)

    score_cols = [c for c in out.columns if c.startswith("s_")]

    # 欠損置換 + 欠損ペナルティ
    missing_count = out[score_cols].isna().sum(axis=1)
    out[score_cols] = out[score_cols].fillna(50.0)
    out["missing_score_items"] = missing_count

    # カテゴリスコア
    out["score_profitability"] = out[["s_op_margin", "s_net_margin", "s_roe"]].mean(axis=1)
    out["score_growth"] = out[["s_revenue_yoy", "s_net_income_yoy"]].mean(axis=1)
    out["score_balance"] = out[["s_debt_to_equity", "s_equity_ratio", "s_roa"]].mean(axis=1)
    out["score_cashflow"] = out[["s_free_cf_margin", "s_operating_cf_margin"]].mean(axis=1)
    out["score_valuation"] = out[["s_pe", "s_pb", "s_ps"]].mean(axis=1)

    out["score_raw"] = (
        out["score_profitability"] * WEIGHTS["profitability"]
        + out["score_growth"] * WEIGHTS["growth"]
        + out["score_balance"] * WEIGHTS["balance"]
        + out["score_cashflow"] * WEIGHTS["cashflow"]
        + out["score_valuation"] * WEIGHTS["valuation"]
    )

    out["missing_penalty"] = out["missing_score_items"] * MISSING_PENALTY_PER_ITEM
    out["score_final"] = (out["score_raw"] - out["missing_penalty"]).clip(lower=0, upper=100)

    # 理由（上位に来た根拠を短く可視化）
    # 上位カテゴリを2つ表示
    cats = ["score_profitability", "score_growth", "score_balance", "score_cashflow", "score_valuation"]
    def top2_reason(row):
        pairs = sorted([(c, row[c]) for c in cats], key=lambda x: x[1], reverse=True)[:2]
        return ",".join([f"{p[0].replace('score_','')}={p[1]:.1f}" for p in pairs])
    out["top_strengths"] = out.apply(top2_reason, axis=1)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="financials_wide.csv path")
    ap.add_argument("--out_dir", default=".", help="output directory")
    ap.add_argument("--top_n", type=int, default=20, help="top N to show in console")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # 企業キーは ticker を基本に
    if "ticker" not in df.columns:
        raise ValueError("ticker column not found in input.")

    # ハード除外
    remain, excluded = hard_exclusion(df)

    # スコア
    scored = scoring(remain)

    # 並べ替え
    scored = scored.sort_values(["score_final", "market_cap"], ascending=[False, False])

    # 出力
    out_screened = f"{args.out_dir.rstrip('/')}/screened_2_5.csv"
    out_excluded = f"{args.out_dir.rstrip('/')}/excluded_2_5.csv"
    out_summary = f"{args.out_dir.rstrip('/')}/screening_2_5_summary.json"

    scored.to_csv(out_screened, index=False)
    excluded.to_csv(out_excluded, index=False)

    summary = {
        "input_rows": int(len(df)),
        "excluded_rows": int(len(excluded)),
        "screened_rows": int(len(scored)),
        "top_n_preview": int(args.top_n),
        "thresholds": HARD_THRESHOLDS,
        "weights": WEIGHTS,
        "missing_penalty_per_item": MISSING_PENALTY_PER_ITEM,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # コンソール表示（上位）
    cols_preview = [
        "ticker", "gx_company_name", "gx_industry",
        "market_cap", "revenue", "net_income",
        "debt_to_equity", "op_margin", "net_margin",
        "score_final", "top_strengths", "missing_score_items"
    ]
    cols_preview = [c for c in cols_preview if c in scored.columns]
    print(scored[cols_preview].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()