#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_eegs_pipeline_batch.py

Batch runner for EEGS collection (≈180 companies).

Reads:
  outputs/interim/repdiv1_presence.csv
required columns:
  - spEmitCode
  - preferred_repDivID   (can be NaN/blank; fallback to 1)

Then calls the single-company pipeline for each company and writes outputs
into an isolated run directory under outputs/interim/ so you never overwrite.

Example
-------
cd /Users/shou/hobby/CPX/nikkei-stock/_20251230_eegs_parse_test
source .venv/bin/activate

# first 5 (debug)
python3 scripts/run_eegs_pipeline_batch.py --limit 5 --save-raw

# continue from 6th item (index=5) in the SAME run dir
python3 scripts/run_eegs_pipeline_batch.py --run-name eegs_batch_YYYYMMDD_HHMMSS --start 5 --save-raw

# full run (new run dir)
python3 scripts/run_eegs_pipeline_batch.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def detect_single_script() -> Path:
    """Pick the most likely single-company pipeline script path."""
    candidates = [
        Path("scripts/eegs_single_company_pipeline.py"),
        Path("scripts/parse_eegs_graph_single.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def ensure_run_dir(base: Path, run_name: Optional[str]) -> Path:
    """Create or reuse run dir under base."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"eegs_batch_{ts}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_targets(path: Path) -> List[Dict[str, Any]]:
    """Read repdiv1_presence.csv (aligned master) and return rows."""
    import pandas as pd

    df = pd.read_csv(path)

    required = ["spEmitCode", "preferred_repDivID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}. Got columns={list(df.columns)}"
        )

    df = df.copy()
    df["spEmitCode"] = df["spEmitCode"].astype(str).str.strip()

    # Keep preferred_repDivID as-is (may be NaN); we coerce safely later
    df = df[df["spEmitCode"].str.len() > 0]
    return df.to_dict(orient="records")


def coerce_repdivid(rep_raw: Any) -> int:
    """Convert preferred_repDivID to int safely. Fallback to 1 for NaN/blank."""
    # NaN check without pandas
    if rep_raw is None:
        return 1
    if isinstance(rep_raw, float) and rep_raw != rep_raw:  # NaN
        return 1

    s = str(rep_raw).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 1

    # Some CSV may contain "1.0" etc.
    try:
        return int(float(s))
    except Exception:
        return 1


def run_one(
    python_exe: str,
    single_script_abs: Path,
    run_dir: Path,
    sp_emit_code: str,
    rep_div_id: int,
    max_gasid: int,
    sleep_s: float,
    save_raw: bool,
) -> Tuple[int, str]:
    """Run single-company pipeline in isolated cwd. Returns (returncode, stderr_tail)."""

    # IMPORTANT: This ordering matches your single script CLI.
    # Global options MUST come before the subcommand.
    cmd = [
        python_exe,
        str(single_script_abs),
        "--spemitcode",
        sp_emit_code,
        "--repdivid",
        str(rep_div_id),
        "pipeline",
        "--max-gasid",
        str(max_gasid),
        "--sleep",
        str(sleep_s),
    ]
    if save_raw:
        cmd.append("--save-raw")

    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    err = (proc.stderr or "").strip()
    tail = err[-2000:] if err else ""
    return proc.returncode, tail


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write manifest.csv (overwrites, but keeps full rows list passed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "spEmitCode",
        "repDivID",
        "status",
        "seconds",
        "single_script",
        "run_outputs_root",
        "error_tail",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch runner for EEGS pipeline")
    p.add_argument(
        "--input-csv",
        default="outputs/interim/repdiv1_presence.csv",
        help="Aligned target table (default: outputs/interim/repdiv1_presence.csv)",
    )
    p.add_argument(
        "--single-script",
        default=None,
        help="Path to single-company pipeline script (default: auto-detect)",
    )
    p.add_argument(
        "--out-base",
        default="outputs/interim",
        help="Base directory to create run directory under (default: outputs/interim)",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Explicit run directory name to reuse/continue (default: eegs_batch_<timestamp>)",
    )
    p.add_argument("--limit", type=int, default=None, help="Process only first N companies (debug)")
    p.add_argument("--start", type=int, default=0, help="Start index (debug/resume)")
    p.add_argument("--max-gasid", type=int, default=80, help="Max gasID to probe (default: 80)")
    p.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds per request inside single pipeline")
    p.add_argument("--save-raw", action="store_true", help="Save raw HTML fragments")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input not found: {input_csv}")

    single_script = Path(args.single_script) if args.single_script else detect_single_script()
    if not single_script.exists():
        raise SystemExit(
            "Single-company script not found. Provide --single-script. Tried: " + str(single_script)
        )

    single_script_abs = single_script.resolve()
    run_dir = ensure_run_dir(Path(args.out_base), args.run_name)

    targets_all = read_targets(input_csv)

    # Apply start/limit
    targets = targets_all
    if args.start:
        targets = targets[args.start:]
    if args.limit is not None:
        targets = targets[: args.limit]

    print("[INFO] input:", input_csv)
    print("[INFO] single_script:", single_script_abs)
    print("[INFO] run_dir:", run_dir)
    print("[INFO] companies:", len(targets))

    python_exe = sys.executable
    manifest_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(targets, start=args.start):
        sp = str(row.get("spEmitCode", "")).strip()
        rep_raw = row.get("preferred_repDivID")
        rep = coerce_repdivid(rep_raw)

        t0 = time.time()
        rc, err_tail = run_one(
            python_exe=python_exe,
            single_script_abs=single_script_abs,
            run_dir=run_dir,
            sp_emit_code=sp,
            rep_div_id=rep,
            max_gasid=args.max_gasid,
            sleep_s=args.sleep,
            save_raw=args.save_raw,
        )
        dt = round(time.time() - t0, 3)

        status = "ok" if rc == 0 else f"error({rc})"
        manifest_rows.append(
            {
                "spEmitCode": sp,
                "repDivID": rep,
                "status": status,
                "seconds": dt,
                "single_script": str(single_script_abs),
                "run_outputs_root": str(run_dir / "outputs"),
                "error_tail": err_tail,
            }
        )

        print(f"[{status}] {idx}: spEmitCode={sp} repDivID={rep} seconds={dt}")
        if rc != 0 and err_tail:
            print("  --- stderr tail ---")
            print("  " + "\n  ".join(err_tail.splitlines()[-10:]))

        # Incremental manifest for resilience
        write_manifest(run_dir / "manifest.csv", manifest_rows)

    print("[DONE] manifest:", run_dir / "manifest.csv")
    print("[DONE] outputs are under:", run_dir / "outputs")


if __name__ == "__main__":
    main()