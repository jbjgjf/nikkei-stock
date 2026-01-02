import pandas as pd
from pathlib import Path

BASE = Path("/Users/shou/hobby/CPX/nikkei-stock")

# 入力（固定化した EEGS long）
INPUT_DIR = BASE / "data/interim/eegs/20251230_batch_20251230_174041/eegs_long"

# 出力（スコアリング run）
OUT_DIR = BASE / "data/interim/eegs_scoring/20251231_gxscore_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = OUT_DIR / "eegs_panel.parquet"
OUT_CSV = OUT_DIR / "eegs_panel.csv"

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ---- 想定カラムの正規化（存在するものだけ使う） ----
    rename_map = {
        "spEmitCode": "spEmitCode",
        "repDivID": "repDivID",
        "year": "year",
        "gasID": "gasID",
        "gasName": "gasName",
        "value": "value",
        "emission": "value",  # 万一別名の場合
        "unit": "unit",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 必須カラムチェック
    required = ["spEmitCode", "repDivID", "year", "gasID", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    # 型の正規化
    df["spEmitCode"] = df["spEmitCode"].astype(str)
    df["repDivID"] = df["repDivID"].astype(int)
    df["year"] = df["year"].astype(int)
    df["gasID"] = df["gasID"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 監査用
    df["source_file"] = path.name

    keep_cols = ["spEmitCode", "repDivID", "year", "gasID", "gasName", "value", "unit", "source_file"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]

def main():
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No csv files found in {INPUT_DIR}")

    frames = []
    for p in files:
        frames.append(load_one(p))

    panel = pd.concat(frames, ignore_index=True)

    # 軽い品質チェック
    panel = panel.dropna(subset=["value"])
    panel = panel.sort_values(["spEmitCode", "year", "gasID"])

    # 保存（parquet優先。未インストールならCSVにフォールバック）
    saved_path = None
    try:
        panel.to_parquet(OUT_PARQUET, index=False)
        saved_path = OUT_PARQUET
    except ImportError:
        panel.to_csv(OUT_CSV, index=False, encoding="utf-8")
        saved_path = OUT_CSV
        print("[WARN] parquet engine not found (pyarrow/fastparquet). Saved CSV instead.")
        print("[HINT] Install one of these to enable parquet:")
        print("       python3 -m pip install pyarrow")
        print("       python3 -m pip install fastparquet")

    # ログ出力
    print("Saved:", saved_path)
    print("Companies:", panel["spEmitCode"].nunique())
    print("Years:", panel["year"].min(), "-", panel["year"].max())
    print("gasIDs:", sorted(panel["gasID"].unique().tolist()))

if __name__ == "__main__":
    main()