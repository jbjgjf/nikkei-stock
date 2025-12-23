#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase C: Fetch EEGS corporate graph JSON for EXACT matches only.

Inputs:
  - screened_2_5_with_envlink.csv (output of Phase B)
Outputs:
  - _generated_envgraph_YYYYMMDD_HHMMSS/ (raw json, long csv, logs, summary)

Notes:
  - EXACT + score==1.0 + single spEmitCode only (no ';')
  - Missing/empty payloads are logged as "missing" (important signal)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests

EEGS_GRAPH_ENDPOINT = "https://eegs.env.go.jp/ghg-santeikohyo-result/corporate/graph"
EEGS_CORP_PAGE = "https://eegs.env.go.jp/ghg-santeikohyo-result/corporate"

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\.ぁ-んァ-ン一-龥]", "_", str(s))
    return s[:120] if len(s) > 120 else s

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_envlink_csv", required=True, help="screened_2_5_with_envlink.csv path")
    ap.add_argument("--out_root", default=None, help="root dir to create timestamp folder under")
    ap.add_argument("--rep_div_id", type=int, default=1, help="repDivID (usually 1)")
    ap.add_argument("--gas_ids", default="auto", help="Comma list like '1,2,3' or 'auto'")
    ap.add_argument("--auto_gas_spEmitCode", default=None,
                    help="If gas_ids=auto, use this spEmitCode to discover gasIDs from HTML. If omitted, first target is used.")
    ap.add_argument("--sleep", type=float, default=0.6, help="sleep seconds between requests")
    ap.add_argument("--timeout", type=float, default=30.0, help="requests timeout seconds")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--max_companies", type=int, default=0, help="0 means no limit")
    ap.add_argument("--cookie", default="", help="Optional Cookie header string (if needed)")
    return ap.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def discover_gas_ids_by_html(spEmitCode: str, repDivID: int, timeout: float, cookie: str) -> List[int]:
    """
    The corporate page HTML typically contains a select box under
    '温室効果ガス算定排出量推移' with options whose values correspond to gasID.
    We scrape option values from the page HTML.
    """
    url = f"{EEGS_CORP_PAGE}?spEmitCode={spEmitCode}&repDivID={repDivID}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if cookie:
        headers["Cookie"] = cookie

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    # option value="1" ... を拾う（ページ内に複数selectがある想定なので広く拾って後でユニーク）
    vals = re.findall(r'<option[^>]+value="(\d+)"', html)
    ids = sorted({int(v) for v in vals})

    # 「全部拾いすぎ」の可能性があるので、graphが実際に通るものに軽く絞る
    # 先頭から試して成功したものだけ残す（最大30個まで）
    keep = []
    for gid in ids[:50]:
        ok, _ = fetch_graph_json(spEmitCode, repDivID, gid, timeout, cookie)
        if ok:
            keep.append(gid)
        if len(keep) >= 30:
            break
    return keep

def fetch_graph_json(spEmitCode: str, repDivID: int, gasID: int, timeout: float, cookie: str) -> Tuple[bool, Dict[str, Any]]:
    params = {"gasID": gasID, "spEmitCode": spEmitCode, "repDivID": repDivID}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{EEGS_CORP_PAGE}?spEmitCode={spEmitCode}&repDivID={repDivID}",
    }
    if cookie:
        headers["Cookie"] = cookie

    try:
        r = requests.get(EEGS_GRAPH_ENDPOINT, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return False, {"status_code": r.status_code, "text": r.text[:500]}
        # EEGSはJSON（またはJSONっぽい）で返す想定
        try:
            data = r.json()
        except Exception:
            # まれに text が返る場合に備える
            return False, {"status_code": r.status_code, "text": r.text[:800]}
        # 空構造は missing として扱う
        if data is None or data == {}:
            return False, {"status_code": 200, "empty": True}
        return True, data
    except Exception as e:
        return False, {"error": repr(e)}

def classify_payload(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Return (status, detail) without raising.

    Status values:
      - HAS_DATA: at least one non-empty numeric datapoint exists
      - SINGLE_YEAR: exactly one year/datapoint exists
      - EMPTY_SERIES: series exists but all data arrays are empty
      - NO_YEARS: cannot find year/category labels
      - NO_DATA_MESSAGE: payload contains a human message indicating no data
      - UNEXPECTED: payload shape not recognized
    """
    if not isinstance(payload, dict):
        return "UNEXPECTED", "payload_not_dict"

    # common message keys
    for k in ["message", "msg", "errorMessage", "notice"]:
        if k in payload and isinstance(payload[k], str) and payload[k].strip():
            return "NO_DATA_MESSAGE", payload[k].strip()[:200]

    years = None
    for k in ["years", "x", "labels", "categories"]:
        if k in payload and isinstance(payload[k], list):
            years = payload[k]
            break

    series = payload.get("series")
    # if series is list of dicts with data
    if isinstance(series, list):
        any_data = False
        total_points = 0
        empty_series = True
        for s in series:
            if isinstance(s, dict) and "data" in s and isinstance(s["data"], list):
                if len(s["data"]) > 0:
                    empty_series = False
                    # count non-null points
                    for v in s["data"]:
                        if v is None:
                            continue
                        # allow numeric-like strings
                        try:
                            float(v)
                            any_data = True
                            total_points += 1
                        except Exception:
                            continue
            else:
                # series element not in expected shape
                empty_series = False

        if any_data:
            if total_points == 1:
                return "SINGLE_YEAR", "only_one_datapoint"
            return "HAS_DATA", f"points={total_points}"
        if empty_series:
            return "EMPTY_SERIES", "series_present_but_empty"

    if years is None or (isinstance(years, list) and len(years) == 0):
        return "NO_YEARS", "no_year_labels"

    return "UNEXPECTED", "no_numeric_points_detected"

def normalize_graph_payload(company_name: str, spEmitCode: str, repDivID: int, gasID: int, payload: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str]:
    """
    payload構造はサイト側都合で変わる可能性があるため、できるだけ柔らかく読む。
    典型的には years + values のような構造を想定し、推定でlong化する。
    """
    status, detail = classify_payload(payload)
    # 候補キーを広めに見る
    years = None
    values = None

    for k in ["years", "x", "labels", "categories"]:
        if k in payload and isinstance(payload[k], list):
            years = payload[k]
            break

    for k in ["values", "y", "data", "series"]:
        if k in payload:
            values = payload[k]
            break

    rows = []
    # ケース1: values が list で years と同長
    if isinstance(years, list) and isinstance(values, list) and len(years) == len(values):
        for y, v in zip(years, values):
            rows.append({"year": y, "value": v})
    # ケース2: series 形式っぽい（[{name, data:[...]}]）
    elif isinstance(values, list) and values and isinstance(values[0], dict) and "data" in values[0] and isinstance(values[0]["data"], list):
        # seriesごとに出す（nameも保持）
        for s in values:
            sname = s.get("name", "")
            sdata = s.get("data", [])
            if isinstance(years, list) and len(years) == len(sdata):
                for y, v in zip(years, sdata):
                    rows.append({"year": y, "value": v, "series": sname})
            else:
                for i, v in enumerate(sdata):
                    rows.append({"year": years[i] if isinstance(years, list) and i < len(years) else i, "value": v, "series": sname})
    else:
        # 構造が読めない場合は空DF（missing扱いにせず“要調査”へ）
        return pd.DataFrame(columns=["company_name","spEmitCode","repDivID","gasID","series","year","value","raw_payload"]), status, detail

    df = pd.DataFrame(rows)
    df["company_name"] = company_name
    df["spEmitCode"] = spEmitCode
    df["repDivID"] = repDivID
    df["gasID"] = gasID
    if "series" not in df.columns:
        df["series"] = ""
    df["raw_payload"] = ""  # 必要なら別途保持
    # If we successfully parsed rows, override status to HAS_DATA unless it is SINGLE_YEAR
    if len(df) == 1:
        status = "SINGLE_YEAR"
        detail = "parsed_one_row"
    else:
        status = "HAS_DATA"
        detail = f"parsed_rows={len(df)}"

    return df[["company_name","spEmitCode","repDivID","gasID","series","year","value","raw_payload"]], status, detail

def main():
    args = parse_args()

    in_csv = args.in_envlink_csv
    out_root = args.out_root
    if out_root is None:
        out_root = os.path.dirname(in_csv)

    outdir = os.path.join(out_root, f"_generated_envgraph_{now_tag()}")
    raw_dir = os.path.join(outdir, "raw_json")
    ensure_dir(outdir)
    ensure_dir(raw_dir)

    df = pd.read_csv(in_csv)

    # === EXACT対象抽出（=2を内包）===
    # Accept either full envlink CSV or exact_targets.csv
    targets = df.copy()
    if "env_match_status" in targets.columns and "env_match_score" in targets.columns:
        # full envlink CSV path
        for col in ["env_match_status", "env_match_score", "env_spEmitCode"]:
            if col not in targets.columns:
                raise ValueError(f"Missing required column: {col}")
        targets = targets[
            (targets["env_match_status"].astype(str) == "EXACT") &
            (targets["env_match_score"].fillna(0).astype(float) >= 1.0) &
            (targets["env_spEmitCode"].notna())
        ].copy()
    else:
        # exact_targets.csv path (already filtered)
        if "env_spEmitCode" not in targets.columns:
            raise ValueError("Input CSV must contain env_spEmitCode")
        print("[INFO] Input looks like exact_targets.csv; skipping EXACT filter", flush=True)

    targets["env_spEmitCode"] = targets["env_spEmitCode"].astype(str)
    targets = targets[~targets["env_spEmitCode"].str.contains(";")].copy()

    if args.max_companies and args.max_companies > 0:
        targets = targets.head(args.max_companies).copy()

    exact_targets_path = os.path.join(outdir, "exact_targets.csv")
    targets.to_csv(exact_targets_path, index=False, encoding="utf-8-sig")

    if len(targets) == 0:
        print("[ERROR] No EXACT targets found under current rules.")
        print(f"[HINT] Check {exact_targets_path} and your input columns.")
        sys.exit(2)

    # gasIDs
    if args.gas_ids.strip().lower() == "auto":
        sample_sp = args.auto_gas_spEmitCode or str(targets["env_spEmitCode"].iloc[0])
        gas_ids = discover_gas_ids_by_html(sample_sp, args.rep_div_id, args.timeout, args.cookie)
        if not gas_ids:
            # 最後の保険：最低限1だけ試す
            gas_ids = [1]
        print(f"[INFO] gas_ids(auto) discovered={gas_ids} using spEmitCode={sample_sp}", flush=True)
    else:
        gas_ids = [int(x) for x in args.gas_ids.split(",") if x.strip().isdigit()]
        if not gas_ids:
            raise ValueError("gas_ids is empty. Use --gas_ids auto or provide like 1,2,3")

    missing_rows = []
    long_frames = []
    ok_count = 0

    n = len(targets)
    for i, row in enumerate(targets.itertuples(index=False), start=1):
        # 会社名列の候補（どれか存在するはず）
        company_name = None
        for c in ["gx_company_name", "company_name", "normalized_name", "name"]:
            if hasattr(row, c):
                company_name = getattr(row, c)
                break
        if company_name is None:
            company_name = "UNKNOWN"

        sp = str(getattr(row, "env_spEmitCode"))
        rep = int(args.rep_div_id)

        for gid in gas_ids:
            ok, payload = fetch_graph_json(sp, rep, gid, args.timeout, args.cookie)
            fn = f"{safe_filename(company_name)}__{sp}__rep{rep}__gas{gid}.json"
            fpath = os.path.join(raw_dir, fn)

            # raw保存（ok/NG問わず）
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            if not ok:
                missing_rows.append({
                    "company_name": company_name,
                    "spEmitCode": sp,
                    "repDivID": rep,
                    "gasID": gid,
                    "reason": payload.get("error") or payload.get("status_code") or "unknown",
                    "detail": (payload.get("text") or "")[:300],
                    "raw_json_path": fpath
                })
                continue

            # long化 + 状態分類
            gdf, p_status, p_detail = normalize_graph_payload(company_name, sp, rep, gid, payload)
            if gdf.empty:
                missing_rows.append({
                    "company_name": company_name,
                    "spEmitCode": sp,
                    "repDivID": rep,
                    "gasID": gid,
                    "status": p_status,
                    "reason": "no_time_series_rows",
                    "detail": p_detail,
                    "raw_json_path": fpath
                })
            else:
                long_frames.append(gdf)
                ok_count += 1
                missing_rows.append({
                    "company_name": company_name,
                    "spEmitCode": sp,
                    "repDivID": rep,
                    "gasID": gid,
                    "status": p_status,
                    "reason": "ok",
                    "detail": p_detail,
                    "raw_json_path": fpath
                })

            time.sleep(args.sleep)

        if (i % args.log_every) == 0 or i == n:
            print(f"[PROGRESS] {i}/{n} companies processed | ok_graphs={ok_count} missing={len(missing_rows)}", flush=True)

        if (i % args.checkpoint_every) == 0 or i == n:
            # checkpoint
            if long_frames:
                long_df = pd.concat(long_frames, ignore_index=True)
                long_df.to_csv(os.path.join(outdir, "graph_long_partial.csv"), index=False, encoding="utf-8-sig")
            pd.DataFrame(missing_rows).to_csv(os.path.join(outdir, "graph_status_partial.csv"), index=False, encoding="utf-8-sig")
            print(f"[CHECKPOINT] saved partials at {i}/{n}", flush=True)

    # final outputs
    long_df = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    long_path = os.path.join(outdir, "graph_long.csv")
    miss_path = os.path.join(outdir, "graph_status.csv")

    if not long_df.empty:
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(miss_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_envlink_csv": in_csv,
        "outdir": outdir,
        "targets_exact": int(n),
        "gas_ids": gas_ids,
        "ok_graph_count": int(ok_count),
        "missing_count": int(len(missing_rows)),
        "long_rows": int(len(long_df)) if not long_df.empty else 0
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Phase C (graph fetch) complete", flush=True)
    print("[OK] outdir:", outdir, flush=True)
    print("[OK] exact_targets.csv:", exact_targets_path, "rows=", n, flush=True)
    print("[OK] graph_long.csv:", long_path, "rows=", summary["long_rows"], flush=True)
    print("[OK] graph_status.csv:", miss_path, "rows=", summary["missing_count"], flush=True)
    print("[OK] summary.json:", os.path.join(outdir, "summary.json"), flush=True)

if __name__ == "__main__":
    main()