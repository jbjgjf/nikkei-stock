#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_eegs_graph_single.py

Purpose
-------
Single-company EEGS collector for one (spEmitCode, repDivID).

It supports:
- fetch: fetch ONE series (gasID) and save the raw HTML fragment.
- probe: discover available gasIDs for the company (early-stop by dropdown count) and write a probe CSV.
- pipeline: (recommended) probe + (optionally) save raw + write ONE long-format CSV (company × metric × year).

Why this design?
---------------
On EEGS, `gasID` is an internal ID and is NOT guaranteed to match dropdown order.
There can be missing IDs and reordering. Therefore we:
- infer the dropdown item count from the corporate page HTML (best-effort)
- probe gasIDs until we collect that many *distinct series names*
- explode each series into long format, which is the most robust base table for later screening.

Inputs
------
Default target:
- spEmitCode = 985336900
- repDivID   = 1

Outputs
-------
fetch mode:
  outputs/raw/eegs_html_single/
    eegs_html_single_spEmitCode=985336900_repDivID=1_gasID=<G>.html

probe mode:
  outputs/interim/eegs_probe/
    eegs_probe_spEmitCode=985336900_repDivID=1.csv

pipeline mode (main deliverable):
  outputs/interim/eegs_long/
    eegs_long_spEmitCode=985336900_repDivID=1.csv
  (also writes probe CSV; and optionally stores raw HTML under outputs/raw/eegs_html_single/)

Run
---
cd /Users/shou/hobby/CPX/nikkei-stock/_20251230_eegs_parse_test

# 1) fetch one gasID (debug)
python3 scripts/parse_eegs_graph_single.py fetch --gasid 9

# 2) probe available gasIDs (debug)
python3 scripts/parse_eegs_graph_single.py probe --max-gasid 80 --sleep 0.25 --save-raw

# 3) end-to-end (recommended): probe + long output
python3 scripts/parse_eegs_graph_single.py pipeline --max-gasid 80 --sleep 0.25 --save-raw
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


CORPORATE_URL = "https://eegs.env.go.jp/ghg-santeikohyo-result/corporate"
GRAPH_URL = "https://eegs.env.go.jp/ghg-santeikohyo-result/corporate/graph"


@dataclass
class GraphSeries:
    gasID: int
    gas_name_raw: str
    gas_name_norm: str
    unit: str
    years: List[int]
    labels: List[str]
    values: List[float]
    ok: bool
    note: str


def normalize_gas_name(s: str) -> str:
    """Remove simple HTML tags/subscripts and normalize whitespace."""
    s = re.sub(r"<\s*sub\s*>", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*sub\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_graph_json(raw_html: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON payload embedded in <script id='graph'>...</script>."""
    m = re.search(
        r"<script\s+type=\"application/json\"\s+id=\"graph\">(.*?)</script>",
        raw_html,
        flags=re.DOTALL,
    )
    if not m:
        return None
    payload = html.unescape(m.group(1)).strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def infer_dropdown_count(corporate_html: str) -> int:
    """Best-effort inference of how many items are in the dropdown.

    Heuristic: count <option> tags inside the first <select>.
    Returns 0 if inference fails.
    """
    m = re.search(r"<select[^>]*>(.*?)</select>", corporate_html, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return 0
    select_inner = m.group(1)
    options = re.findall(r"<option[^>]*>(.*?)</option>", select_inner, flags=re.DOTALL | re.IGNORECASE)
    cleaned = [re.sub(r"<[^>]+>", "", o).strip() for o in options]
    cleaned = [c for c in cleaned if c]
    return len(cleaned)


def fetch(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int = 30) -> str:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_corporate_page(sp_emit_code: str, rep_div_id: int, headers: Dict[str, str]) -> str:
    return fetch(CORPORATE_URL, params={"spEmitCode": sp_emit_code, "repDivID": rep_div_id}, headers=headers)


def fetch_graph_page(sp_emit_code: str, rep_div_id: int, gas_id: int, headers: Dict[str, str]) -> str:
    return fetch(
        GRAPH_URL,
        params={"spEmitCode": sp_emit_code, "repDivID": rep_div_id, "gasID": gas_id},
        headers=headers,
    )


def parse_series(gas_id: int, raw_html: str) -> GraphSeries:
    obj = extract_graph_json(raw_html)
    if obj is None:
        return GraphSeries(
            gasID=gas_id,
            gas_name_raw="",
            gas_name_norm="",
            unit="",
            years=[],
            labels=[],
            values=[],
            ok=False,
            note="no_graph_json",
        )

    years = obj.get("repYear") or []
    values = obj.get("emitAmount") or []
    labels = obj.get("label") or []
    unit = obj.get("unit") or ""
    gas_name_raw = obj.get("gasName") or ""

    if not isinstance(years, list) or not isinstance(values, list) or not years:
        return GraphSeries(
            gasID=gas_id,
            gas_name_raw=str(gas_name_raw),
            gas_name_norm=normalize_gas_name(str(gas_name_raw)),
            unit=str(unit),
            years=[],
            labels=[],
            values=[],
            ok=False,
            note="empty_or_invalid_years",
        )

    # Cast
    years_i: List[int] = []
    try:
        years_i = [int(y) for y in years]
    except Exception:
        years_i = []

    values_f: List[float] = []
    for v in values:
        try:
            values_f.append(float(v))
        except Exception:
            values_f.append(float("nan"))

    labels_s = [str(x) for x in labels] if isinstance(labels, list) else []

    return GraphSeries(
        gasID=gas_id,
        gas_name_raw=str(gas_name_raw),
        gas_name_norm=normalize_gas_name(str(gas_name_raw)),
        unit=str(unit),
        years=years_i,
        labels=labels_s,
        values=values_f,
        ok=True,
        note="ok",
    )


def series_to_long_rows(sp_emit_code: str, rep_div_id: int, s: GraphSeries, fetched_at_iso: str) -> List[Dict[str, Any]]:
    """Explode one GraphSeries into long rows.

    Rule: if label is '報告なし', treat as not-reported and set value to empty (NaN in pandas later).
    """
    rows: List[Dict[str, Any]] = []

    # Align lengths defensively
    n = len(s.years)
    n = min(n, len(s.values))
    if s.labels:
        n = min(n, len(s.labels))

    for i in range(n):
        year = s.years[i]
        val = s.values[i]
        label = s.labels[i] if s.labels else ""
        is_reported = 1
        if label == "報告なし":
            is_reported = 0
            # represent missing as empty string in CSV (pandas will read as NaN if na_values configured)
            val_out: Any = ""
        else:
            val_out = val

        rows.append(
            {
                "spEmitCode": sp_emit_code,
                "repDivID": rep_div_id,
                "gasID": s.gasID,
                "gas_name_raw": s.gas_name_raw,
                "gas_name_norm": s.gas_name_norm,
                "unit": s.unit,
                "year": year,
                "value": val_out,
                "label": label,
                "is_reported": is_reported,
                "fetched_at": fetched_at_iso,
                "source_url": f"{GRAPH_URL}?gasID={s.gasID}&spEmitCode={sp_emit_code}&repDivID={rep_div_id}",
            }
        )

    return rows


def write_probe_csv(out_csv: Path, sp_emit_code: str, rep_div_id: int, found: Dict[int, GraphSeries]) -> None:
    rows: List[Dict[str, Any]] = []
    for gas_id, s in found.items():
        rows.append(
            {
                "spEmitCode": sp_emit_code,
                "repDivID": rep_div_id,
                "gasID": gas_id,
                "ok": int(s.ok),
                "note": s.note,
                "gas_name_raw": s.gas_name_raw,
                "gas_name_norm": s.gas_name_norm,
                "unit": s.unit,
                "years_count": len(s.years),
                "years": ",".join(str(y) for y in s.years),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spEmitCode",
                "repDivID",
                "gasID",
                "ok",
                "note",
                "gas_name_raw",
                "gas_name_norm",
                "unit",
                "years_count",
                "years",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_long_csv(out_csv: Path, long_rows: List[Dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spEmitCode",
                "repDivID",
                "gasID",
                "gas_name_raw",
                "gas_name_norm",
                "unit",
                "year",
                "value",
                "label",
                "is_reported",
                "fetched_at",
                "source_url",
            ],
        )
        w.writeheader()
        w.writerows(long_rows)


def run_probe(
    sp_emit_code: str,
    rep_div_id: int,
    max_gasid: int,
    sleep_s: float,
    jitter_s: float,
    stop_after_misses: int,
    save_raw: bool,
) -> Dict[str, Any]:
    """Probe gasIDs for a single company and return details.

    Returns dict with keys:
    - dropdown_n
    - found (gasID -> GraphSeries)
    - found_by_name (gas_name_norm -> gasID)
    - raw_dir (Path)
    """
    headers = {"User-Agent": "Mozilla/5.0"}

    corp_html = fetch_corporate_page(sp_emit_code, rep_div_id, headers=headers)
    dropdown_n = infer_dropdown_count(corp_html)

    raw_dir = Path("outputs/raw/eegs_html_single")
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    found: Dict[int, GraphSeries] = {}
    found_by_name: Dict[str, int] = {}
    consecutive_miss = 0

    def should_stop() -> bool:
        if dropdown_n <= 0:
            return False
        return len(found_by_name) >= dropdown_n

    print("[INFO] inferred dropdown count (best-effort):", dropdown_n)
    if dropdown_n == 0:
        print("[WARN] could not infer dropdown count from HTML. Will probe up to max_gasid.")

    for gas_id in range(0, max_gasid + 1):
        raw_html = ""
        try:
            raw_html = fetch_graph_page(sp_emit_code, rep_div_id, gas_id, headers=headers)
            series = parse_series(gas_id, raw_html)
        except requests.RequestException as e:
            series = GraphSeries(
                gasID=gas_id,
                gas_name_raw="",
                gas_name_norm="",
                unit="",
                years=[],
                labels=[],
                values=[],
                ok=False,
                note=f"request_error:{type(e).__name__}",
            )

        if save_raw and raw_html:
            raw_path = raw_dir / f"eegs_html_single_spEmitCode={sp_emit_code}_repDivID={rep_div_id}_gasID={gas_id}.html"
            raw_path.write_text(raw_html, encoding="utf-8")

        found[gas_id] = series

        if series.ok and series.gas_name_norm:
            consecutive_miss = 0
            found_by_name.setdefault(series.gas_name_norm, gas_id)
            print(f"[OK] gasID={gas_id} name='{series.gas_name_norm}' years={len(series.years)}")
        else:
            consecutive_miss += 1

        if should_stop():
            print("[STOP] reached inferred dropdown count:", dropdown_n)
            break

        if dropdown_n == 0 and consecutive_miss >= stop_after_misses:
            print(
                f"[STOP] dropdown unknown; {consecutive_miss} consecutive misses reached stop_after_misses={stop_after_misses}"
            )
            break

        if sleep_s > 0:
            time.sleep(sleep_s + random.uniform(0, jitter_s))

    return {
        "dropdown_n": dropdown_n,
        "found": found,
        "found_by_name": found_by_name,
        "raw_dir": raw_dir,
    }


def cmd_fetch(args: argparse.Namespace) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    raw_html = fetch_graph_page(args.spemitcode, args.repdivid, args.gasid, headers=headers)

    out_dir = Path("outputs/raw/eegs_html_single")
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"eegs_html_single_spEmitCode={args.spemitcode}_repDivID={args.repdivid}_gasID={args.gasid}.html"
    out_path = out_dir / fname
    out_path.write_text(raw_html, encoding="utf-8")

    series = parse_series(args.gasid, raw_html)

    print("[OK] saved raw EEGS graph response to:")
    print(out_path)
    print("\n--- Parsed summary ---")
    print(f"gasID={series.gasID} ok={series.ok} note={series.note}")
    print(f"gas_name_norm={series.gas_name_norm}")
    if series.ok:
        print(f"years={series.years}")
        print(f"labels={series.labels}")
        print(f"values(head)={series.values[:5]}")


def cmd_probe(args: argparse.Namespace) -> None:
    out_probe_dir = Path("outputs/interim/eegs_probe")
    out_probe_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_probe_dir / f"eegs_probe_spEmitCode={args.spemitcode}_repDivID={args.repdivid}.csv"

    result = run_probe(
        sp_emit_code=args.spemitcode,
        rep_div_id=args.repdivid,
        max_gasid=args.max_gasid,
        sleep_s=args.sleep,
        jitter_s=args.jitter,
        stop_after_misses=args.stop_after_misses,
        save_raw=args.save_raw,
    )

    write_probe_csv(out_csv, args.spemitcode, args.repdivid, result["found"])
    print("[OK] wrote:", out_csv)
    print("[SUMMARY] ok_series_by_name=", len(result["found_by_name"]))


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Probe and write ONE long CSV for a single company."""
    fetched_at_iso = datetime.now(timezone.utc).isoformat()

    # 1) probe
    out_probe_dir = Path("outputs/interim/eegs_probe")
    out_probe_csv = out_probe_dir / f"eegs_probe_spEmitCode={args.spemitcode}_repDivID={args.repdivid}.csv"

    result = run_probe(
        sp_emit_code=args.spemitcode,
        rep_div_id=args.repdivid,
        max_gasid=args.max_gasid,
        sleep_s=args.sleep,
        jitter_s=args.jitter,
        stop_after_misses=args.stop_after_misses,
        save_raw=args.save_raw,
    )

    write_probe_csv(out_probe_csv, args.spemitcode, args.repdivid, result["found"])
    print("[OK] wrote:", out_probe_csv)

    # 2) build long rows from *successful* distinct series names
    found: Dict[int, GraphSeries] = result["found"]
    found_by_name: Dict[str, int] = result["found_by_name"]

    long_rows: List[Dict[str, Any]] = []
    for name, gas_id in sorted(found_by_name.items(), key=lambda x: x[1]):
        s = found.get(gas_id)
        if not s or not s.ok:
            continue
        long_rows.extend(series_to_long_rows(args.spemitcode, args.repdivid, s, fetched_at_iso=fetched_at_iso))

    out_long_dir = Path("outputs/interim/eegs_long")
    out_long_csv = out_long_dir / f"eegs_long_spEmitCode={args.spemitcode}_repDivID={args.repdivid}.csv"
    write_long_csv(out_long_csv, long_rows)

    print("[OK] wrote:", out_long_csv)
    print("[SUMMARY] series=", len(found_by_name), "rows=", len(long_rows))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EEGS single-company collector")
    p.add_argument("--spemitcode", default="985336900", help="EEGS spEmitCode (default: 985336900)")
    p.add_argument("--repdivid", type=int, default=1, help="EEGS repDivID (default: 1)")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch one gasID and save raw HTML")
    p_fetch.add_argument("--gasid", type=int, default=9, help="gasID to fetch (default: 9)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_probe = sub.add_parser("probe", help="Probe gasIDs until reaching dropdown count (early-stop)")
    p_probe.add_argument("--max-gasid", type=int, default=80, help="Max gasID to try (default: 80)")
    p_probe.add_argument("--sleep", type=float, default=0.25, help="Base sleep between requests (seconds)")
    p_probe.add_argument("--jitter", type=float, default=0.15, help="Random jitter added to sleep (seconds)")
    p_probe.add_argument(
        "--stop-after-misses",
        type=int,
        default=25,
        help="If dropdown count cannot be inferred, stop after this many consecutive misses",
    )
    p_probe.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save each probed raw HTML response under outputs/raw/eegs_html_single/",
    )
    p_probe.set_defaults(func=cmd_probe)

    p_pipe = sub.add_parser("pipeline", help="Probe + write one long CSV (recommended)")
    p_pipe.add_argument("--max-gasid", type=int, default=80, help="Max gasID to try (default: 80)")
    p_pipe.add_argument("--sleep", type=float, default=0.25, help="Base sleep between requests (seconds)")
    p_pipe.add_argument("--jitter", type=float, default=0.15, help="Random jitter added to sleep (seconds)")
    p_pipe.add_argument(
        "--stop-after-misses",
        type=int,
        default=25,
        help="If dropdown count cannot be inferred, stop after this many consecutive misses",
    )
    p_pipe.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save each probed raw HTML response under outputs/raw/eegs_html_single/",
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()