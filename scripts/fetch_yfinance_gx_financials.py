import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


BASE = Path(__file__).resolve().parents[1]

# 入力（あなたの253社：1/2だけ + 手動修正済みの前提）
IN_PATH = BASE / "data/data_collection/processed/gxleague_tickers_with_rank_1_2.csv"

# 出力フォルダ（毎回新規）
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = BASE / f"data/data_collection/processed/yfinance_gx_{RUN_TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WIDE = OUT_DIR / "financials_wide.csv"
OUT_LONG = OUT_DIR / "financials_long_annual.csv"
OUT_ERR = OUT_DIR / "errors.csv"
OUT_SUM = OUT_DIR / "coverage_summary.json"


# ==========
# Utilities
# ==========
def to_yf_symbol(ticker: str) -> str:
    """TSE tickers need .T. Accept already-suffixed tickers."""
    t = str(ticker).strip()
    if "." in t:
        return t
    return f"{t}.T"


def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        # pandas scalar
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return np.nan


def first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance financial tables are (rows=item names, cols=dates).
    Convert to tidy long: (item, asof, value).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["item", "asof", "value"])
    out = (
        df.copy()
        .reset_index()
        .melt(id_vars=["index"], var_name="asof", value_name="value")
        .rename(columns={"index": "item"})
    )
    # ensure datetime-friendly
    out["asof"] = pd.to_datetime(out["asof"], errors="coerce")
    return out


def pick_latest(df: pd.DataFrame) -> tuple[pd.Timestamp | None, dict]:
    """Return latest column date and dict(item->value) for that column."""
    if df is None or df.empty:
        return None, {}
    latest = df.columns[0]
    series = df[latest]
    return pd.to_datetime(latest, errors="coerce"), series.to_dict()


def compute_growth(latest: float, prev: float) -> float:
    if np.isnan(latest) or np.isnan(prev) or prev == 0:
        return np.nan
    return (latest / prev) - 1.0


def compute_margin(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


# ==========
# Main
# ==========
def main():
    src = pd.read_csv(IN_PATH)

    # Safety: remove duplicate tickers
    if "ticker" not in src.columns:
        raise ValueError(f"Input must include 'ticker' column: {IN_PATH}")

    src = src.drop_duplicates(subset=["ticker"]).copy()

    print(f"[START] input={IN_PATH}")
    print(f"[START] output_dir={OUT_DIR}")
    print(f"[START] total_tickers={len(src)}")

    wide_rows = []
    long_rows = []
    err_rows = []

    # coverage counters
    c_ok = 0
    c_info = 0
    c_is = 0
    c_bs = 0
    c_cf = 0

    for idx, r in src.iterrows():
        ticker = str(r["ticker"]).strip()
        yf_symbol = to_yf_symbol(ticker)

        base = {
            "ticker": ticker,
            "yf_symbol": yf_symbol,
            "gx_company_name": r.get("gx_company_name"),
            "mapped_company_name": r.get("mapped_company_name"),
            "emission_label": r.get("emission_label"),
            "gx_industry": r.get("業種"),
            "rank_company_name": r.get("会社名"),
        }

        try:
            t = yf.Ticker(yf_symbol)

            # --- meta (valuation-ish) ---
            # Use fast_info first (lighter), then info fallback (heavier)
            fi = {}
            try:
                fi = t.fast_info or {}
            except Exception:
                fi = {}

            # Some valuation fields exist in info (may fail; keep optional)
            info = {}
            try:
                # Only call if we actually need extra fields (kept small)
                info = t.get_info() or {}
                c_info += 1
            except Exception:
                info = {}

            market_cap = safe_float(first_present(fi, ["market_cap"]) or first_present(info, ["marketCap"]))
            last_price = safe_float(first_present(fi, ["last_price"]) or first_present(info, ["currentPrice"]))
            shares = safe_float(first_present(fi, ["shares"]) or first_present(info, ["sharesOutstanding"]))

            # --- statements (annual) ---
            fin_is = t.financials
            fin_bs = t.balance_sheet
            fin_cf = t.cashflow

            if fin_is is not None and not fin_is.empty:
                c_is += 1
            if fin_bs is not None and not fin_bs.empty:
                c_bs += 1
            if fin_cf is not None and not fin_cf.empty:
                c_cf += 1

            # Latest + previous for growth
            is_asof, is_latest = pick_latest(fin_is)
            bs_asof, bs_latest = pick_latest(fin_bs)
            cf_asof, cf_latest = pick_latest(fin_cf)

            # previous year (2nd column)
            is_prev = {}
            is_prev_asof = None
            if fin_is is not None and not fin_is.empty and len(fin_is.columns) >= 2:
                is_prev_asof = pd.to_datetime(fin_is.columns[1], errors="coerce")
                is_prev = fin_is[fin_is.columns[1]].to_dict()

            # Key line items (yfinance uses English labels)
            revenue = safe_float(first_present(is_latest, ["Total Revenue", "TotalRevenue"]))
            gross_profit = safe_float(first_present(is_latest, ["Gross Profit", "GrossProfit"]))
            op_income = safe_float(first_present(is_latest, ["Operating Income", "OperatingIncome"]))
            pretax_income = safe_float(first_present(is_latest, ["Pretax Income", "PretaxIncome"]))
            net_income = safe_float(first_present(is_latest, ["Net Income", "NetIncome"]))

            revenue_prev = safe_float(first_present(is_prev, ["Total Revenue", "TotalRevenue"]))
            net_income_prev = safe_float(first_present(is_prev, ["Net Income", "NetIncome"]))

            total_assets = safe_float(first_present(bs_latest, ["Total Assets", "TotalAssets"]))
            total_equity = safe_float(first_present(bs_latest, ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity"]))
            total_liab = safe_float(first_present(bs_latest, ["Total Liab", "TotalLiab"]))

            # Debt-ish (best-effort)
            total_debt = safe_float(first_present(bs_latest, ["Total Debt", "TotalDebt"]))
            if np.isnan(total_debt):
                # sometimes only short/long term
                st_debt = safe_float(first_present(bs_latest, ["Short Long Term Debt", "ShortLongTermDebt", "Short Term Debt", "ShortTermDebt"]))
                lt_debt = safe_float(first_present(bs_latest, ["Long Term Debt", "LongTermDebt"]))
                total_debt = np.nansum([st_debt, lt_debt]) if (not np.isnan(st_debt) or not np.isnan(lt_debt)) else np.nan

            op_cf = safe_float(first_present(cf_latest, ["Total Cash From Operating Activities", "TotalCashFromOperatingActivities"]))
            capex = safe_float(first_present(cf_latest, ["Capital Expenditures", "CapitalExpenditures"]))
            free_cf = safe_float(first_present(cf_latest, ["Free Cash Flow", "FreeCashFlow"]))
            if np.isnan(free_cf) and (not np.isnan(op_cf) or not np.isnan(capex)):
                free_cf = op_cf + capex  # capex is usually negative in statements

            # --- Derived metrics (2) ---
            # Profitability
            net_margin = compute_margin(net_income, revenue)
            op_margin = compute_margin(op_income, revenue)

            # ROE / ROA
            roe = compute_margin(net_income, total_equity)
            roa = compute_margin(net_income, total_assets)

            # Leverage
            debt_to_equity = compute_margin(total_debt, total_equity)

            # Growth (YoY using 2 latest annual columns)
            revenue_yoy = compute_growth(revenue, revenue_prev)
            net_income_yoy = compute_growth(net_income, net_income_prev)

            # Valuation (best-effort)
            pe = safe_float(first_present(info, ["trailingPE", "forwardPE"]))
            pb = safe_float(first_present(info, ["priceToBook"]))
            # If info missing, compute rough P/S from market cap & revenue
            ps = np.nan
            if not np.isnan(market_cap) and not np.isnan(revenue) and revenue != 0:
                ps = market_cap / revenue

            wide = {
                **base,
                "asof_is": str(is_asof.date()) if is_asof is not None and not pd.isna(is_asof) else None,
                "asof_bs": str(bs_asof.date()) if bs_asof is not None and not pd.isna(bs_asof) else None,
                "asof_cf": str(cf_asof.date()) if cf_asof is not None and not pd.isna(cf_asof) else None,

                # (1) absolute financials
                "revenue": revenue,
                "gross_profit": gross_profit,
                "operating_income": op_income,
                "pretax_income": pretax_income,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liab,
                "total_equity": total_equity,
                "total_debt": total_debt,
                "operating_cf": op_cf,
                "capex": capex,
                "free_cf": free_cf,

                # (2) valuation & ratios
                "market_cap": market_cap,
                "last_price": last_price,
                "shares_outstanding": shares,
                "pe": pe,
                "pb": pb,
                "ps_est": ps,
                "net_margin": net_margin,
                "op_margin": op_margin,
                "roe": roe,
                "roa": roa,
                "debt_to_equity": debt_to_equity,
                "revenue_yoy": revenue_yoy,
                "net_income_yoy": net_income_yoy,
            }
            wide_rows.append(wide)

            # long format (annual) for IS/BS/CF
            # tag statement type and ticker
            for stmt_name, df in [("income_statement", fin_is), ("balance_sheet", fin_bs), ("cashflow", fin_cf)]:
                tidy = normalize_financial_df(df)
                if not tidy.empty:
                    tidy["ticker"] = ticker
                    tidy["yf_symbol"] = yf_symbol
                    tidy["statement"] = stmt_name
                    tidy["emission_label"] = r.get("emission_label")
                    tidy["gx_company_name"] = r.get("gx_company_name")
                    long_rows.append(tidy)

            c_ok += 1

        except Exception as e:
            err_rows.append({**base, "error": repr(e)})

        # progress log (毎銘柄)
        print(
            f"[PROGRESS] {idx+1}/{len(src)} {yf_symbol} ok={len(wide_rows)} err={len(err_rows)}",
            flush=True,
        )

        # periodic partial saves (実行中にファイルが見える & 中断しても途中結果が残る)
        if (idx + 1) % 10 == 0:
            pd.DataFrame(wide_rows).to_csv(
                OUT_DIR / "financials_wide_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(err_rows).to_csv(
                OUT_DIR / "errors_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"[CHECKPOINT] saved partials at {idx+1}/{len(src)}",
                flush=True,
            )
            time.sleep(2.0)
        else:
            time.sleep(0.35)

    wide_df = pd.DataFrame(wide_rows)
    err_df = pd.DataFrame(err_rows)
    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
    else:
        long_df = pd.DataFrame(columns=["ticker", "yf_symbol", "statement", "item", "asof", "value"])

    # save
    wide_df.to_csv(OUT_WIDE, index=False, encoding="utf-8-sig")
    long_df.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    err_df.to_csv(OUT_ERR, index=False, encoding="utf-8-sig")

    summary = {
        "input_rows": int(len(src)),
        "success_rows": int(c_ok),
        "error_rows": int(len(err_df)),
        "has_info_calls": int(c_info),
        "income_statement_available": int(c_is),
        "balance_sheet_available": int(c_bs),
        "cashflow_available": int(c_cf),
        "output_dir": str(OUT_DIR),
        "files": {
            "wide": str(OUT_WIDE),
            "long": str(OUT_LONG),
            "errors": str(OUT_ERR),
        },
    }
    OUT_SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Output folder:", OUT_DIR)
    print("[OK] Wide CSV:", OUT_WIDE, "rows=", len(wide_df))
    print("[OK] Long CSV:", OUT_LONG, "rows=", len(long_df))
    print("[INFO] Errors:", OUT_ERR, "rows=", len(err_df))
    print("[INFO] Summary:", OUT_SUM)


if __name__ == "__main__":
    main()