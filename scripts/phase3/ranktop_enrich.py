#!/usr/bin/env python3
"""
ranktop_enrich.py

Final portfolio builder for Phase3.

This script:
1. Integrates three fixed inputs:
   - phase3_ranktop_full.csv (NLP-based Phase3 scores)
   - fromnotion.csv (manual/Notion-based qualitative info)
   - patent_features_company.csv (J-PlatPat-based patent aggregates)
2. Resolves company name variations via normalized company_key.
3. Applies optional categorical-to-numeric mappings (score_mapper.yaml).
4. Computes MCDA-based final scores and ranks.
5. Outputs a single CSV that represents the FINAL portfolio ranking.

The output CSV alone is sufficient to:
- reproduce the ranking,
- explain the score decomposition,
- extract the Top-20 companies deterministically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3 Ranktop Enrichment + MCDA Final Ranking")
    p.add_argument("--ranktop_full_csv", type=Path, required=True)
    p.add_argument("--fromnotion_csv", type=Path, required=True)
    p.add_argument("--patent_features_company_csv", type=Path, required=True)
    p.add_argument("--score_mapper_yaml", type=Path, default=None)
    p.add_argument("--out_csv", type=Path, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: list[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}")


_PAREN_RE = re.compile(r"\([^)]*\)|（[^）]*）")
_LEAD_CORP_RE = re.compile(r"^(株式会社|（株）|\(株\))")
_TRAIL_CORP_RE = re.compile(r"(株式会社)$")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    s = str(name) if name is not None else ""
    s = s.strip().replace("\u3000", " ")
    s = _MULTI_SPACE_RE.sub(" ", s)

    if s.endswith("_removed"):
        s = s[:-8].strip()

    s = _PAREN_RE.sub("", s).strip()
    s = _LEAD_CORP_RE.sub("", s).strip()
    s = _TRAIL_CORP_RE.sub("", s).strip()
    s = _MULTI_SPACE_RE.sub(" ", s)

    return s


def _add_company_key(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
    out = df.copy()
    out[src_col] = out[src_col].astype(str)
    out["company_key"] = out[src_col].apply(normalize_company_name)
    return out


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    mn, mx = s.min(), s.max()
    if mx - mn == 0:
        return pd.Series(0.0, index=series.index)
    return (s - mn) / (mx - mn)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    out = n / d
    return out.where(d.notna() & (d != 0))


# ---------------------------------------------------------------------
# Score mapper
# ---------------------------------------------------------------------
def _load_score_mapper(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_score_mapper(df: pd.DataFrame, mapper: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    mappings = mapper.get("mappings", {})
    for col, mp in mappings.items():
        if col not in out.columns:
            continue

        def _map(x):
            if pd.isna(x):
                return pd.NA
            return mp.get(str(x).strip(), pd.NA)

        out[col + "_num"] = out[col].apply(_map)
    return out


# ---------------------------------------------------------------------
# MCDA computation
# ---------------------------------------------------------------------
def compute_mcda(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # =========================
    # A. Phase3 (NLP)
    # =========================
    out["phase3_core_ratio"] = _safe_ratio(out.get("core_yes"), out.get("core_den")).fillna(0)

    axes = [
        "structure_coverage",
        "impl_strength",
        "transparency_strength",
        "supplychain_strength",
        "transition_strength",
        "innovation_strength",
        "regulation_strength",
    ]
    for c in axes:
        out[c + "_norm"] = _minmax(out[c]) if c in out.columns else 0.0

    out["phase3_score"] = (
        0.35 * out["phase3_core_ratio"]
        + 0.20 * out["impl_strength_norm"]
        + 0.15 * out["structure_coverage_norm"]
        + 0.10 * out["transparency_strength_norm"]
        + 0.10 * out["supplychain_strength_norm"]
        + 0.05 * out["transition_strength_norm"]
        + 0.05 * out["innovation_strength_norm"]
    ).clip(0, 1)

    # =========================
    # B. Notion
    # =========================
    notion_cols = [
        "notion__sbti_joined_num",
        "notion__scope12_disclosure_num",
        "notion__scope3_disclosure_num",
        "notion__icp_status_num",
        "notion__voluntary_credit_usage_num",
    ]
    available = [c for c in notion_cols if c in out.columns]
    out["notion_score"] = out[available].mean(axis=1, skipna=True).fillna(0)

    # =========================
    # C. Patents
    # =========================
    apps5y = pd.to_numeric(out.get("patent__apps_count_5y"), errors="coerce").fillna(0)
    total = pd.to_numeric(out.get("patent__doc_count_total"), errors="coerce").fillna(0)

    activity = np.log1p(apps5y) + 0.3 * np.log1p(total)
    strength = (
        0.5 * pd.to_numeric(out.get("patent__grant_rate"), errors="coerce").fillna(0)
        + 0.3 * pd.to_numeric(out.get("patent__valid_patent_rate"), errors="coerce").fillna(0)
        + 0.2 * pd.to_numeric(out.get("patent__maintenance_event_rate"), errors="coerce").fillna(0)
    )
    breadth = np.log1p(pd.to_numeric(out.get("patent__fi_unique_count"), errors="coerce").fillna(0))

    out["patent_score"] = (
        0.4 * _minmax(activity)
        + 0.4 * _minmax(strength)
        + 0.2 * _minmax(breadth)
    ).fillna(0)

    # =========================
    # D. Credibility
    # =========================
    nm = out.get("notion_matched", False).astype(bool)
    pm = out.get("patent_matched", False).astype(bool)
    penalty = 0.5 * (~nm).astype(int) + 0.5 * (~pm).astype(int)
    survey = pd.to_numeric(out.get("notion__survey_answered_num"), errors="coerce").fillna(0)

    out["credibility_score"] = (1 - penalty + 0.2 * survey).clip(0, 1)

    # =========================
    # Final MCDA
    # =========================
    out["mcda_A"] = (0.6 * out["phase3_score"] + 0.4 * out["notion_score"]).clip(0, 1)
    out["mcda_B"] = out["patent_score"].clip(0, 1)
    out["mcda_C"] = out["credibility_score"].clip(0, 1)

    out["final_score"] = (
        0.45 * out["mcda_A"]
        + 0.35 * out["mcda_B"]
        + 0.20 * out["mcda_C"]
    )

    out["final_rank"] = out["final_score"].rank(ascending=False, method="dense").astype(int)
    out["final_top20_flag"] = out["final_rank"] <= 20

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    rank = pd.read_csv(args.ranktop_full_csv)
    notion = pd.read_csv(args.fromnotion_csv)
    patents = pd.read_csv(args.patent_features_company_csv)

    _require_columns(rank, ["entity_id"], "ranktop")
    _require_columns(notion, ["company_name"], "notion")
    _require_columns(patents, ["company_name"], "patents")

    rank = _add_company_key(rank, "entity_id")
    notion = _add_company_key(notion, "company_name")
    patents = _add_company_key(patents, "company_name")

    notion = notion.drop_duplicates("company_key", keep="last")
    patents = patents.drop_duplicates("company_key", keep="last")

    notion = notion.rename(columns={c: f"notion__{c}" for c in notion.columns if c != "company_key"})
    patents = patents.rename(columns={c: f"patent__{c}" for c in patents.columns if c != "company_key"})

    merged = rank.merge(notion, on="company_key", how="left")
    merged["notion_matched"] = merged["notion__company_name"].notna()

    merged = merged.merge(patents, on="company_key", how="left")
    merged["patent_matched"] = merged["patent__company_name"].notna()

    if args.score_mapper_yaml:
        mapper = _load_score_mapper(args.score_mapper_yaml)
        merged = _apply_score_mapper(merged, mapper)

    merged = compute_mcda(merged)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    print("[OK] Final portfolio written to:", args.out_csv)
    print("[INFO] Top 5 preview:")
    print(
        merged.sort_values("final_score", ascending=False)
        .head(5)[["final_rank", "entity_id", "company_key", "final_score"]]
        .assign(final_score=lambda d: d["final_score"].round(4))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()