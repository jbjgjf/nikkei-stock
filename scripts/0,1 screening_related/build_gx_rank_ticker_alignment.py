import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

RANK_PATH = BASE / "data/data_collection/processed/GXleague_rank.csv"
TICK_PATH = BASE / "data/data_collection/processed/gxleague_tickers.csv"

OUT_PATH = BASE / "data/data_collection/processed/gxleague_tickers_with_rank_1_2.csv"
UNMATCHED_PATH = BASE / "data/data_collection/processed/gxleague_tickers_unmatched_in_rank.csv"
DUP_RANK_PATH = BASE / "data/data_collection/processed/GXleague_rank_duplicates_by_normname.csv"


def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).replace("　", " ").strip()
    # 法人格・空白・注記を除去（必要に応じて増やす）
    s = (s.replace("株式会社", "")
           .replace("（株）", "")
           .replace("(株)", "")
           .replace("有限会社", "")
           .replace("合同会社", "")
           .replace(" ", "")
           .replace("※", ""))
    return s


def main():
    rank = pd.read_csv(RANK_PATH)
    tick = pd.read_csv(TICK_PATH)

    # 1) ラベル列は「排出量区分」を正として扱う（あなたの実データ仕様）
    if "排出量区分" not in rank.columns:
        raise ValueError(f"GXleague_rank.csv に '排出量区分' 列がありません。columns={list(rank.columns)}")

    # 2) 正規化キー
    rank["name_norm"] = rank["会社名"].map(norm_name)
    tick["name_norm"] = tick["gx_company_name"].map(norm_name)

    # 3) rank側の重複監査（m:1を成立させるために必須）
    dups = rank[rank.duplicated("name_norm", keep=False)].sort_values("name_norm")
    if len(dups) > 0:
        dups.to_csv(DUP_RANK_PATH, index=False, encoding="utf-8-sig")
        print(f"[WARN] rank側に正規化社名の重複があります: {DUP_RANK_PATH} (rows={len(dups)})")
    else:
        print("[OK] rank側の正規化社名は一意です")

    # 4) rank側を一意化（暫定ルール：同一name_normは先頭行を採用）
    #    ※もし「同一社名でラベルが違う」などが見つかったら、優先ルールを決めてここを変更する
    rank_unique = (rank.sort_values(["name_norm"])
                       .drop_duplicates("name_norm", keep="first")
                       .rename(columns={"排出量区分": "emission_label"}))

    # 5) 整合（tickers ← rank を left join：tickerを落とさずラベルを付与）
    merged = tick.merge(
        rank_unique[["name_norm", "会社名", "業種", "emission_label"]],
        on="name_norm",
        how="left",
        validate="m:1"
    )

    # 6) 1/2だけ抽出して保存
    merged_12 = merged[merged["emission_label"].isin([1, 2])].copy()
    merged_12.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {OUT_PATH} (rows={len(merged_12)})")

    # 7) マッチしなかったものも保存（品質確認用）
    unmatched = merged[merged["emission_label"].isna()].copy()
    unmatched[["gx_company_name", "ticker", "mapped_company_name"]].to_csv(
        UNMATCHED_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"[INFO] Saved unmatched: {UNMATCHED_PATH} (rows={len(unmatched)})")

    # 8) サマリ
    print("\n--- SUMMARY ---")
    print("tickers rows:", len(tick))
    print("matched any label:", merged["emission_label"].notna().sum())
    print("matched label in {1,2}:", len(merged_12))


if __name__ == "__main__":
    main()