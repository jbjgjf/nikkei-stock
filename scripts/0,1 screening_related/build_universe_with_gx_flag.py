import pandas as pd
from pathlib import Path

# Data base directory (you said you manage data under src/data_collection)
BASE = Path("src/data_collection")
RAW = BASE / "raw"
PROC = BASE / "processed"

# Candidate locations for inputs (pick the first one that exists)
JPX_CANDIDATES = [
    RAW / "jpx_company_ticker_map.csv",
    PROC / "jpx_company_ticker_map.csv",
    BASE / "jpx_company_ticker_map.csv",
    Path("data/raw/jpx_company_ticker_map.csv"),
    Path("data/processed/jpx_company_ticker_map.csv"),
]

GX_CANDIDATES = [
    PROC / "gxleague_tickers.csv",
    BASE / "gxleague_tickers.csv",
    Path("data/processed/gxleague_tickers.csv"),
]

PROC.mkdir(parents=True, exist_ok=True)


def pick_existing(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def main() -> None:
    jpx_path = pick_existing(JPX_CANDIDATES, "JPX company→ticker map (jpx_company_ticker_map.csv)")
    gx_path = pick_existing(GX_CANDIDATES, "GX tickers (gxleague_tickers.csv)")

    # 1) 全上場企業（母集団）
    jpx = pd.read_csv(jpx_path, dtype=str)
    if "ticker" not in jpx.columns:
        raise ValueError(f"'ticker' column missing in {jpx_path}. Columns: {list(jpx.columns)}")
    jpx["ticker"] = jpx["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    # 2) GX League 上場企業
    gx = pd.read_csv(gx_path, dtype=str)
    if "ticker" not in gx.columns:
        raise ValueError(f"'ticker' column missing in {gx_path}. Columns: {list(gx.columns)}")
    gx["ticker"] = gx["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    # GX側は ticker だけ使う
    gx_flag = gx[["ticker"]].drop_duplicates().copy()
    gx_flag["is_gx"] = 1

    # 3) left join でフラグ付与
    universe = jpx.merge(gx_flag, on="ticker", how="left")
    universe["is_gx"] = universe["is_gx"].fillna(0).astype(int)

    # 4) 保存
    out_path = PROC / "universe_with_gx_flag.csv"
    universe.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("JPX map path:", jpx_path)
    print("GX tickers path:", gx_path)
    print("Universe size:", len(universe))
    print("GX companies :", int(universe["is_gx"].sum()))
    print("Non-GX       :", int((universe["is_gx"] == 0).sum()))


if __name__ == "__main__":
    main()