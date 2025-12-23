import re
import json
import csv
from pathlib import Path

import requests

# --- Config (edit as needed) ---
SP_EMIT_CODE = "985336900"   # e.g. Nippon Steel
REP_DIV_ID = "1"

BASE = "https://eegs.env.go.jp"
CORP_URL = f"{BASE}/ghg-santeikohyo-result/corporate?spEmitCode={SP_EMIT_CODE}&repDivID={REP_DIV_ID}"
GRAPH_URL_TMPL = f"{BASE}/ghg-santeikohyo-result/graph?gasID={{gas_id}}&spEmitCode={SP_EMIT_CODE}&repDivID={REP_DIV_ID}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": f"{BASE}/",
}


def fetch_corporate_html(sess: requests.Session) -> str:
    r = sess.get(CORP_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def discover_gas_ids(html: str):
    """Try to discover gasID values used by the dropdown/graphs.

    We prefer parsing <option value="..."> if present.
    If not found, we fall back to looking for occurrences like 'gasID=NN'.
    """
    # option values (most stable)
    option_ids = re.findall(r"<option[^>]+value=\"(\d+)\"", html)
    option_ids = [int(x) for x in option_ids]

    if option_ids:
        # de-dup preserving order
        seen = set()
        out = []
        for x in option_ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # fallback: any gasID occurrences
    ids = re.findall(r"gasID=(\d+)", html)
    ids = [int(x) for x in ids]
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def fetch_graph_json(sess: requests.Session, gas_id: int):
    url = GRAPH_URL_TMPL.format(gas_id=gas_id)
    r = sess.get(url, headers=HEADERS, timeout=30)

    # 404 means this gasID is not available for this company/division.
    # We treat it as a valid "missing" signal, not as a fatal error.
    if r.status_code == 404:
        return None

    r.raise_for_status()

    # Some endpoints return JSON with correct content-type; others return JSON-like text.
    try:
        return r.json()
    except Exception:
        return json.loads(r.text)


def normalize_graph_payload(gas_id: int, payload):
    """Normalize graph payload into rows: year, value, unit/label if available.

    Because the payload schema may change, we handle common patterns:
    - {labels:[...], datasets:[{data:[...], label:"..."}], ...}
    - {x:[...], y:[...]} or similar
    Returns a list[dict].
    """
    rows = []

    if payload is None:
        rows.append({
            "spEmitCode": SP_EMIT_CODE,
            "repDivID": REP_DIV_ID,
            "gasID": gas_id,
            "series": "__missing__",
            "unit": "",
            "year": "",
            "value": "",
        })
        return rows

    # Pattern A: Chart.js-like
    if isinstance(payload, dict) and "labels" in payload and "datasets" in payload:
        labels = payload.get("labels") or []
        datasets = payload.get("datasets") or []
        for ds in datasets:
            data = ds.get("data") or []
            label = ds.get("label") or ""
            unit = ds.get("unit") or payload.get("unit") or ""
            for i, year in enumerate(labels):
                val = data[i] if i < len(data) else None
                rows.append({
                    "spEmitCode": SP_EMIT_CODE,
                    "repDivID": REP_DIV_ID,
                    "gasID": gas_id,
                    "series": label,
                    "unit": unit,
                    "year": year,
                    "value": val,
                })
        return rows

    # Pattern B: x/y arrays
    if isinstance(payload, dict):
        x = payload.get("x") or payload.get("years") or payload.get("labels")
        y = payload.get("y") or payload.get("values") or payload.get("data")
        if isinstance(x, list) and isinstance(y, list):
            unit = payload.get("unit") or ""
            for i, year in enumerate(x):
                val = y[i] if i < len(y) else None
                rows.append({
                    "spEmitCode": SP_EMIT_CODE,
                    "repDivID": REP_DIV_ID,
                    "gasID": gas_id,
                    "series": payload.get("label") or "",
                    "unit": unit,
                    "year": year,
                    "value": val,
                })
            return rows

    # Unknown schema: keep a single row with raw payload for debugging
    rows.append({
        "spEmitCode": SP_EMIT_CODE,
        "repDivID": REP_DIV_ID,
        "gasID": gas_id,
        "series": "",
        "unit": "",
        "year": "",
        "value": "",
        "raw": json.dumps(payload, ensure_ascii=False),
    })
    return rows


def main():
    s = requests.Session()

    # 1) Load corporate page (establish cookies/session)
    html = fetch_corporate_html(s)
    print("[OK] corporate page fetched")

    # 2) Discover gasIDs
    gas_ids = discover_gas_ids(html)
    print(f"[INFO] discovered gasIDs: {gas_ids[:30]}{' ...' if len(gas_ids) > 30 else ''} (n={len(gas_ids)})")

    if not gas_ids:
        print("[WARN] no gasIDs discovered from HTML. Use DevTools Network -> Copy as cURL to confirm the graph endpoint and IDs.")
        return

    # 3) Single test: fetch the first gasID and show a preview
    test_id = gas_ids[0]
    payload = fetch_graph_json(s, test_id)
    if payload is None:
        print(f"[WARN] graph JSON not available for gasID={test_id} (404)")
    else:
        print(f"[OK] graph JSON fetched for gasID={test_id}")
        print("[DEBUG] payload preview:")
        print(json.dumps(payload, ensure_ascii=False)[:800])

    # 4) Optional: fetch ALL gasIDs and save normalized rows to CSV
    out_dir = Path("data/data_collection/processed/eegs_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"eegs_graph_long_spEmitCode_{SP_EMIT_CODE}_repDivID_{REP_DIV_ID}.csv"

    all_rows = []
    for gid in gas_ids:
        try:
            pl = fetch_graph_json(s, gid)
            if pl is None:
                # Record missing indicator as one row considered in downstream features
                rows = [{
                    "spEmitCode": SP_EMIT_CODE,
                    "repDivID": REP_DIV_ID,
                    "gasID": gid,
                    "series": "__missing__",
                    "unit": "",
                    "year": "",
                    "value": "",
                }]
            else:
                rows = normalize_graph_payload(gid, pl)
            all_rows.extend(rows)
            print(f"[PROGRESS] gasID={gid} rows+={len(rows)}")
        except Exception as e:
            print(f"[ERROR] gasID={gid}: {e}")

    # write CSV
    # collect union of keys
    keys = set()
    for r in all_rows:
        keys.update(r.keys())
    keys = [k for k in ["spEmitCode","repDivID","gasID","series","unit","year","value","raw"] if k in keys] + [k for k in sorted(keys) if k not in {"spEmitCode","repDivID","gasID","series","unit","year","value","raw"}]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"[OK] saved: {out_csv} rows={len(all_rows)}")


if __name__ == "__main__":
    main()