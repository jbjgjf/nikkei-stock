#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a JPX ticker → English name map using yfinance (long/short names).

Inputs:
- src/data_collection/processed/universe_with_gx_flag.csv
    (must contain a ticker column)

Outputs:
- src/data_collection/processed/jpx_ticker_english_name.csv
    columns: ticker, yfinance_long_name, yfinance_short_name, english_name_norm,
             is_instrument, instrument_reason, source
- src/data_collection/processed/universe_operating_companies.csv
    filtered map where is_instrument == 0

Notes:
- Tickers are queried as "{ticker}.T" to use the Tokyo exchange suffix.
- Results are cached on disk: existing rows in the output file are kept and
  only missing tickers are fetched to minimize yfinance calls.
- Default limit is 200 to keep yfinance calls practical; override with --limit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf


# -------------------------------------
# Normalization (English company names)
# -------------------------------------
_EN_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "CORPORATION",
    "CORP",
    "CO",
    "CO LTD",
    "CO LIMITED",
    "COMPANY",
    "COMPANY LIMITED",
    "LTD",
    "LIMITED",
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "PLC",
    "AG",
    "SA",
    "NV",
}

_EN_PUNCT = r"[\\s\\.,'\"/&\\-–—_()\\[\\]{}]"

# Instrument classification keywords (uppercased; matched on sanitized names)
_INSTRUMENT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\\bETF\\b"), "ETF keyword"),
    (re.compile(r"\\bETN\\b"), "ETN keyword"),
    (re.compile(r"\\bREIT\\b"), "REIT keyword"),
    (re.compile(r"\\bJ\\s*REIT\\b"), "J-REIT keyword"),
    (re.compile(r"\\bFUND\\b"), "Fund keyword"),
    (re.compile(r"\\bTRUST\\b"), "Trust keyword"),
    (re.compile(r"\\bINDEX\\b"), "Index keyword"),
    (re.compile(r"\\bNIKKEI\\b"), "Index keyword: Nikkei"),
    (re.compile(r"\\bTOPIX\\b"), "Index keyword: TOPIX"),
    (re.compile(r"\\bINVERSE\\b"), "Inverse/short keyword"),
    (re.compile(r"\\bSHORT\\b"), "Inverse/short keyword"),
    (re.compile(r"\\bBEAR\\b"), "Inverse/short keyword"),
    (re.compile(r"\\bLEVERAGE\\w*\\b"), "Leveraged keyword"),
    (re.compile(r"\\bGEARED\\b"), "Leveraged keyword"),
    (re.compile(r"\\bBOND\\b"), "Bond/notes keyword"),
    (re.compile(r"\\bNOTE\\b"), "Bond/notes keyword"),
)


def _strip_en_suffix_tokens(tokens):
    """Remove trailing legal suffix tokens repeatedly."""
    # Normalize tokens that may come from punctuation splits (e.g., "CO", "LTD")
    toks = tokens[:]
    while toks:
        candidate_two = " ".join(toks[-2:]) if len(toks) >= 2 else ""
        if candidate_two and candidate_two in _EN_SUFFIXES:
            toks = toks[:-2]
            continue
        if toks[-1] in _EN_SUFFIXES:
            toks = toks[:-1]
            continue
        break
    return toks


def normalize_english_name(name: str) -> str:
    """Normalize an English company name for matching.

    Steps:
    1) Uppercase
    2) Remove punctuation and whitespace
    3) Strip common corporate suffixes (INC/CORP/CO/LTD/HOLDINGS/GROUP/PLC/etc.)
    4) Remove remaining spaces
    """
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""

    s = s.upper()
    s = s.replace("&", " ")
    s = re.sub(_EN_PUNCT, " ", s)
    tokens = [t for t in s.split() if t]
    tokens = _strip_en_suffix_tokens(tokens)
    return "".join(tokens)


def load_universe(universe_path: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_path, encoding="utf-8-sig")
    # Accept common ticker column names
    ticker_col_candidates = ["ticker", "securities_code", "銘柄コード", "証券コード", "code"]
    ticker_col = next((c for c in ticker_col_candidates if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(f"Missing ticker column in {universe_path}. Columns: {list(df.columns)}")

    df = df.rename(columns={ticker_col: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(4)
    return df


def _sanitize_for_instrument(s: str) -> str:
    """Uppercase and replace punctuation with spaces for instrument matching."""
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


def classify_instrument(long_name: Optional[str], short_name: Optional[str]) -> Tuple[int, str]:
    """Classify whether a ticker represents an instrument (ETF/ETN/REIT/etc.)."""
    for label, raw in (("long", long_name), ("short", short_name)):
        if not raw or pd.isna(raw):
            continue
        text = _sanitize_for_instrument(str(raw))
        for pat, reason in _INSTRUMENT_PATTERNS:
            if pat.search(text):
                return 1, f"{reason} ({label} name)"
    return 0, ""


def fetch_yfinance_names(ticker: str) -> Tuple[Dict[str, Optional[str]], Optional[str]]:
    try:
        yf_ticker = yf.Ticker(f"{ticker}.T")
        info = yf_ticker.get_info() or {}
        return (
            {
                "yfinance_long_name": info.get("longName"),
                "yfinance_short_name": info.get("shortName"),
            },
            None,
        )
    except Exception as e:  # pragma: no cover - defensive
        return {"yfinance_long_name": None, "yfinance_short_name": None}, str(e)


def build_english_map(universe_path: Path, out_path: Path, opco_out_path: Path, limit: Optional[int]) -> pd.DataFrame:
    universe = load_universe(universe_path)
    tickers = universe["ticker"].dropna().unique().tolist()
    if limit and limit > 0:
        tickers = tickers[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opco_out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        if "ticker" in existing.columns:
            existing["ticker"] = existing["ticker"].astype(str).str.strip().str.zfill(4)
        else:
            existing = pd.DataFrame()

    existing_map = {t: row for t, row in existing.set_index("ticker").iterrows()} if not existing.empty else {}

    print(f"[INFO] Total tickers in universe: {len(universe)} (limit applied: {len(tickers)})")
    records = []
    for idx, ticker in enumerate(tickers, 1):
        if ticker in existing_map:
            row = existing_map[ticker]
            long_name = row.get("yfinance_long_name")
            short_name = row.get("yfinance_short_name")

            english_norm = row.get("english_name_norm", "")
            if pd.isna(english_norm) or not english_norm:
                english_norm = normalize_english_name(long_name or short_name or "")
            else:
                english_norm = normalize_english_name(english_norm)

            is_instr_val = row.get("is_instrument") if "is_instrument" in row else None
            is_instr = None
            if is_instr_val is not None and not pd.isna(is_instr_val):
                try:
                    is_instr = int(is_instr_val)
                except Exception:
                    is_instr = None
            instr_reason = row.get("instrument_reason", "") if "instrument_reason" in row else ""
            if is_instr is None:
                is_instr, instr_reason = classify_instrument(long_name, short_name)
            elif pd.isna(instr_reason) or not instr_reason:
                _, instr_reason = classify_instrument(long_name, short_name)

            records.append(
                {
                    "ticker": ticker,
                    "yfinance_long_name": long_name,
                    "yfinance_short_name": short_name,
                    "english_name_norm": english_norm,
                    "is_instrument": int(is_instr),
                    "instrument_reason": instr_reason or "",
                    "source": "cache",
                }
            )
            if idx % 50 == 0:
                print(f"[INFO] Processed {idx}/{len(tickers)} (cache hit)")
            continue

        names, error = fetch_yfinance_names(ticker)
        long_name = names.get("yfinance_long_name")
        short_name = names.get("yfinance_short_name")
        english_norm = normalize_english_name(long_name or short_name or "")
        is_instr, instr_reason = classify_instrument(long_name, short_name)

        if error:
            print(f"[WARN] yfinance fetch failed for {ticker}: {error}")

        records.append(
            {
                "ticker": ticker,
                "yfinance_long_name": long_name,
                "yfinance_short_name": short_name,
                "english_name_norm": english_norm,
                "is_instrument": is_instr,
                "instrument_reason": instr_reason if instr_reason else ("yfinance_error" if error else ""),
                "source": "yfinance",
            }
        )

        if idx % 50 == 0 or idx == len(tickers):
            print(f"[INFO] Processed {idx}/{len(tickers)}")

    result = pd.DataFrame(records)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    opco_df = result[result["is_instrument"] == 0].copy()
    opco_df.to_csv(opco_out_path, index=False, encoding="utf-8-sig")
    print(
        f"[OK] JPX English name map written: {out_path} (rows={len(result)}); "
        f"Operating companies: {len(opco_df)} -> {opco_out_path}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=".",
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--universe",
        default="src/data_collection/processed/universe_with_gx_flag.csv",
        help="Path to universe_with_gx_flag.csv",
    )
    parser.add_argument(
        "--out",
        default="src/data_collection/processed/jpx_ticker_english_name.csv",
        help="Output path for the JPX English name map",
    )
    parser.add_argument(
        "--opco_out",
        default="src/data_collection/processed/universe_operating_companies.csv",
        help="Output path for operating-company universe (is_instrument==0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit number of tickers to fetch (default: 200; set to 0 to disable)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    try:
        build_english_map(
            root / args.universe,
            root / args.out,
            root / args.opco_out,
            args.limit,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
