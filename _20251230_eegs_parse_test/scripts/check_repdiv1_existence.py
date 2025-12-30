

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_repdiv1_existence.py

Purpose
-------
Create an *interim* decision table that tells, for each target company (spEmitCode),
whether EEGS has repDivID=1 available in the national master list.

This is a *consistency/availability* check step only:
- It does NOT rewrite any source CSV.
- It does NOT fetch EEGS pages.
- It only reads from outputs/source and writes to outputs/interim.

Inputs (default)
----------------
1) outputs/source/Japan_eegs.csv
   A master list of EEGS corporate page URLs. One of its columns contains URLs like:
     https://eegs.env.go.jp/ghg-santeikohyo-result/corporate?spEmitCode=985336900&repDivID=1

2) outputs/source/exact_targets.csv
   A target list generated from 2nd screening.
   It should include a column for spEmitCode (case-insensitive).

Outputs
-------
1) outputs/interim/repdiv1_presence.csv
   Columns (minimum set):
     - spEmitCode
     - has_repDiv1
     - repDivIDs (a string like "[1, 5]")
     - preferred_repDivID (1 if available else smallest available repDivID)
     - preferred_url (URL for preferred_repDivID if found)
     - note (diagnostic)

Optional (printed to stdout)
----------------------------
- Summary counts and a small distribution of preferred_repDivID among companies without repDivID=1.

Run
---
From repo root:
  cd /Users/shou/hobby/CPX/nikkei-stock/_20251230_eegs_parse_test
  python scripts/check_repdiv1_existence.py

With custom paths:
  python scripts/check_repdiv1_existence.py --master <path> --targets <path> --out <path>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# --- URL parsing (EEGS corporate page) ---
_RE_SPEMIT = re.compile(r"spEmitCode=([^&#]+)")
_RE_REPDIV = re.compile(r"repDivID=([^&#]+)")


def _parse_spemit_repdiv(url: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract spEmitCode and repDivID from an EEGS corporate URL."""
    if not isinstance(url, str) or not url:
        return None, None
    m_sp = _RE_SPEMIT.search(url)
    m_rd = _RE_REPDIV.search(url)
    sp = m_sp.group(1) if m_sp else None
    rd_raw = m_rd.group(1) if m_rd else None
    try:
        rd = int(rd_raw) if rd_raw is not None else None
    except Exception:
        rd = None
    return sp, rd


def _detect_url_column(df: pd.DataFrame) -> str:
    """Find the URL column in Japan_eegs.csv by name or content."""
    name_candidates = [
        "url",
        "URL",
        "page",
        "Page",
        "link",
        "リンク",
        "ページ",
    ]
    for c in name_candidates:
        if c in df.columns:
            return c

    # content-based heuristic
    best_col: Optional[str] = None
    best_hits = -1
    for c in df.columns:
        s = df[c].astype(str)
        hits = int(s.str.contains("eegs.env.go.jp", na=False).sum())
        if hits > best_hits:
            best_hits = hits
            best_col = c

    if best_col is None or best_hits <= 0:
        raise ValueError(
            "Could not detect the EEGS URL column in Japan_eegs.csv. "
            "Please rename the URL column to one of: url/page/リンク/ページ."
        )
    return best_col


def _detect_spemit_column(df: pd.DataFrame) -> str:
    """Find the spEmitCode column in exact_targets.csv (case-insensitive)."""
    # exact match (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["spemitcode", "sp_emit_code", "sp_emitcode", "spemit"]:
        if key in lower_map:
            return lower_map[key]

    # substring fallback
    for c in df.columns:
        if "spEmitCode" in c or "spemit" in c.lower():
            return c

    raise ValueError(
        "Could not find a spEmitCode column in exact_targets.csv. "
        "Add a column named 'spEmitCode' (recommended) and retry."
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]  # .../_20251230_eegs_parse_test

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master",
        default=str(base_dir / "outputs" / "source" / "Japan_eegs.csv"),
        help="Path to outputs/source/Japan_eegs.csv",
    )
    ap.add_argument(
        "--targets",
        default=str(base_dir / "outputs" / "source" / "exact_targets.csv"),
        help="Path to outputs/source/exact_targets.csv",
    )
    ap.add_argument(
        "--out",
        default=str(base_dir / "outputs" / "interim" / "repdiv1_presence.csv"),
        help="Output path under outputs/interim",
    )
    args = ap.parse_args()

    master_path = Path(args.master)
    targets_path = Path(args.targets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load master ---
    df_master = pd.read_csv(master_path)
    url_col = _detect_url_column(df_master)

    parsed = df_master[url_col].astype(str).apply(_parse_spemit_repdiv)
    df_m = df_master.copy()
    df_m["spEmitCode"] = parsed.apply(lambda x: x[0])
    df_m["repDivID"] = parsed.apply(lambda x: x[1])
    df_m = df_m.dropna(subset=["spEmitCode", "repDivID"]).copy()
    df_m["spEmitCode"] = df_m["spEmitCode"].astype(str)
    df_m["repDivID"] = df_m["repDivID"].astype(int)

    # repDivIDs list per spEmitCode
    rep_list = (
        df_m.groupby("spEmitCode")["repDivID"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
        .reset_index()
        .rename(columns={"repDivID": "repDivIDs"})
    )

    # URL lookup per (spEmitCode, repDivID)
    df_urls = df_m[["spEmitCode", "repDivID", url_col]].copy()
    df_urls = df_urls.rename(columns={url_col: "url"})
    df_urls = df_urls.drop_duplicates(subset=["spEmitCode", "repDivID"], keep="first")

    # --- Load targets ---
    df_targets = pd.read_csv(targets_path)
    sp_col = _detect_spemit_column(df_targets)
    out = df_targets.copy()
    out["spEmitCode"] = out[sp_col].astype(str)

    # merge repDiv list
    out = out.merge(rep_list, on="spEmitCode", how="left")

    # compute presence
    def has1(v) -> bool:
        return isinstance(v, list) and (1 in v)

    def pref_rep(v) -> Optional[int]:
        if isinstance(v, list) and len(v) > 0:
            return 1 if 1 in v else int(v[0])
        return None

    out["has_repDiv1"] = out["repDivIDs"].apply(has1)
    out["preferred_repDivID"] = out["repDivIDs"].apply(pref_rep)

    # attach preferred URL
    out = out.merge(
        df_urls,
        left_on=["spEmitCode", "preferred_repDivID"],
        right_on=["spEmitCode", "repDivID"],
        how="left",
    )
    out = out.drop(columns=["repDivID"], errors="ignore")
    out = out.rename(columns={"url": "preferred_url"})

    # stringify repDivIDs for CSV stability
    out["repDivIDs"] = out["repDivIDs"].apply(lambda v: str(v) if isinstance(v, list) else "")

    # diagnostics
    def note_row(row) -> str:
        if not row.get("repDivIDs"):
            return "missing_in_master"
        if bool(row.get("has_repDiv1")):
            return "ok_repDiv1_available"
        if pd.isna(row.get("preferred_url")):
            return "no_url_for_preferred_repDiv"
        return "repDiv1_missing_use_preferred"

    out["note"] = out.apply(note_row, axis=1)

    # stable column ordering (keep original columns first)
    extra_cols = ["spEmitCode", "has_repDiv1", "repDivIDs", "preferred_repDivID", "preferred_url", "note"]
    cols = list(df_targets.columns) + [c for c in extra_cols if c not in df_targets.columns]
    out = out[cols]

    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --- stdout summary ---
    n_total = len(out)
    n_has1 = int(out["has_repDiv1"].fillna(False).sum())
    n_missing = int((out["note"] == "missing_in_master").sum())
    print(f"[OK] wrote: {out_path}")
    print(f"[SUMMARY] targets={n_total} | has_repDiv1={n_has1} ({(n_has1/n_total if n_total else 0):.1%}) | missing_in_master={n_missing}")

    # lightweight distribution (only for those without repDiv1)
    no1 = out[(out["has_repDiv1"] == False) & (out["preferred_repDivID"].notna())]
    if not no1.empty:
        dist = no1["preferred_repDivID"].value_counts().sort_index()
        print("[SUMMARY] preferred_repDivID for targets without repDivID=1:")
        for k, v in dist.items():
            print(f"  repDivID={int(k)}: {int(v)}")


if __name__ == "__main__":
    main()