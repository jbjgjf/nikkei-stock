#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make a long-format emission table from EEGS raw_json wrappers.

Input files are saved by fetch_env_graph_exact.py under:
  .../_generated_envgraph_YYYYMMDD_HHMMSS/raw_json/*.json

Each file typically looks like:
  {"status_code": 200, "text": "<div>...<script id=\"graph\">{&quot;repYear&quot;:[...], ...}</script>..."}

This script extracts the embedded graph JSON, unescapes it, parses repYear/emitAmount,
then writes a long table suitable for analysis.

Outputs:
  - graph_long_from_raw.csv  (main)
  - graph_status_from_raw.csv (audit)

Author: CPX / nikkei-stock
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_RE = re.compile(r'<script[^>]*id="graph"[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class ParseResult:
    ok: bool
    status: str
    detail: str
    n_rows: int


def parse_filename(meta_name: str) -> Tuple[str, str, int, int]:
    """Parse: 会社名__spEmitCode__rep1__gas1.json -> (company, spEmitCode, repDivID, gasID)"""
    base = os.path.basename(meta_name)
    base = base[:-5] if base.endswith(".json") else base
    parts = base.split("__")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {base}")
    company = parts[0]
    sp = parts[1]
    rep_part = parts[2]
    gas_part = parts[3]

    rep = int(re.sub(r"\D", "", rep_part)) if rep_part else 0
    gas = int(re.sub(r"\D", "", gas_part)) if gas_part else 0
    return company, sp, rep, gas


def extract_graph_dict(wrapper: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (graph_dict, note)."""
    if isinstance(wrapper, dict):
        # Already a graph dict
        if "repYear" in wrapper and "emitAmount" in wrapper:
            return wrapper, "already_graph_dict"

        # Wrapper with HTML in text
        text = wrapper.get("text")
        if isinstance(text, str):
            m = SCRIPT_RE.search(text)
            if not m:
                return None, "no_script_graph"
            raw = html.unescape(m.group(1)).strip()
            try:
                gp = json.loads(raw)
            except Exception as e:
                return None, f"graph_json_load_failed:{type(e).__name__}"
            if not isinstance(gp, dict):
                return None, "graph_json_not_dict"
            return gp, "extracted_from_html"

    return None, "unrecognized_wrapper"


def graph_dict_to_rows(company: str, sp: str, rep: int, gas: int, gp: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    years = gp.get("repYear")
    values = gp.get("emitAmount")

    unit = gp.get("unit", "")
    gas_name = gp.get("gasName", "")
    if isinstance(gas_name, str):
        gas_name = TAG_RE.sub("", gas_name)  # remove <sub> etc.

    if not isinstance(years, list) or not isinstance(values, list):
        return [], "repYear_or_emitAmount_not_list"
    if len(years) == 0:
        return [], "empty_repYear"
    if len(years) != len(values):
        return [], f"length_mismatch years={len(years)} values={len(values)}"

    rows: List[Dict[str, Any]] = []
    bad = 0
    for y, v in zip(years, values):
        try:
            year_i = int(y)
        except Exception:
            bad += 1
            continue
        try:
            val_f = float(v)
        except Exception:
            bad += 1
            continue

        rows.append({
            "company_name": company,
            "spEmitCode": sp,
            "repDivID": rep,
            "gasID": gas,
            "gas_name": gas_name,
            "year": year_i,
            "value": val_f,
            "unit": unit,
        })

    if not rows:
        return [], f"no_numeric_rows bad={bad}"
    return rows, f"ok rows={len(rows)} bad={bad}"


def process_one(path: str) -> Tuple[List[Dict[str, Any]], ParseResult, Dict[str, Any]]:
    company, sp, rep, gas = parse_filename(path)

    try:
        wrapper = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        return [], ParseResult(False, "READ_FAIL", f"{type(e).__name__}", 0), {
            "file": path, "company_name": company, "spEmitCode": sp, "repDivID": rep, "gasID": gas
        }

    gp, note = extract_graph_dict(wrapper)
    if gp is None:
        return [], ParseResult(False, "NO_GRAPH", note, 0), {
            "file": path, "company_name": company, "spEmitCode": sp, "repDivID": rep, "gasID": gas,
            "status_code": wrapper.get("status_code") if isinstance(wrapper, dict) else None,
        }

    rows, detail = graph_dict_to_rows(company, sp, rep, gas, gp)
    if not rows:
        return [], ParseResult(False, "NO_ROWS", detail, 0), {
            "file": path, "company_name": company, "spEmitCode": sp, "repDivID": rep, "gasID": gas,
            "note": note,
        }

    return rows, ParseResult(True, "OK", detail, len(rows)), {
        "file": path, "company_name": company, "spEmitCode": sp, "repDivID": rep, "gasID": gas,
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_json_dir", required=True, help="Folder containing raw_json/*.json")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (default: <raw_json_dir>/../graph_long_from_raw.csv)")
    ap.add_argument("--out_status", default=None, help="Output status CSV path (default: <raw_json_dir>/../graph_status_from_raw.csv)")
    args = ap.parse_args()

    raw_dir = args.raw_json_dir
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_json_dir not found: {raw_dir}")

    out_dir = os.path.abspath(os.path.join(raw_dir, os.pardir))
    out_csv = args.out_csv or os.path.join(out_dir, "graph_long_from_raw.csv")
    out_status = args.out_status or os.path.join(out_dir, "graph_status_from_raw.csv")

    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json")]
    files.sort()

    all_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    ok_files = 0
    for i, fpath in enumerate(files, start=1):
        rows, pres, meta = process_one(fpath)
        all_rows.extend(rows)
        status_rows.append({
            **meta,
            "ok": pres.ok,
            "status": pres.status,
            "detail": pres.detail,
            "n_rows": pres.n_rows,
        })
        if pres.ok:
            ok_files += 1
        if i % 25 == 0 or i == len(files):
            print(f"[PROGRESS] {i}/{len(files)} files | ok_files={ok_files} long_rows={len(all_rows)}", flush=True)

    # Write outputs
    df_long = pd.DataFrame(all_rows)
    if not df_long.empty:
        df_long = df_long.sort_values(["spEmitCode", "gasID", "year"]).reset_index(drop=True)
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")

    df_stat = pd.DataFrame(status_rows)
    df_stat.to_csv(out_status, index=False, encoding="utf-8-sig")

    print("[OK] long:", out_csv, "rows=", len(df_long), flush=True)
    print("[OK] status:", out_status, "rows=", len(df_stat), flush=True)


if __name__ == "__main__":
    main()