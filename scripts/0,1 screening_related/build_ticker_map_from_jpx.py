import pandas as pd
from pathlib import Path
import re

JPX_XLS = "/Users/shou/hobby/CPX/nikkei-stock/src/data_collection/data_j.xls"   # ←あなたのファイル名に合わせる
GX_CSV  = "/Users/shou/hobby/CPX/nikkei-stock/src/data_collection/GXleague_names.csv"

out_raw = Path("data/raw"); out_raw.mkdir(parents=True, exist_ok=True)
out_proc = Path("data/processed"); out_proc.mkdir(parents=True, exist_ok=True)

def norm(s: str) -> str:
    s = str(s).strip().replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(株式会社|（株）|\(株\))", "", s)
    return s.strip()

# 1) JPX Excel → company_name,ticker を作る（列名は実物に合わせて調整）
df = pd.read_excel(JPX_XLS)

# ここが重要：列名をあなたのExcelの列名に合わせる
# よくある列名例：「コード」「銘柄名」
code_col = "コード"
name_col = "銘柄名"

df_map = df[[name_col, code_col]].dropna().copy()
df_map["ticker"] = df_map[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
df_map["company_name"] = df_map[name_col].astype(str)
df_map["company_name_norm"] = df_map["company_name"].map(norm)

map_path = out_raw / "jpx_company_ticker_map.csv"
df_map[["company_name","ticker","company_name_norm"]].to_csv(map_path, index=False, encoding="utf-8-sig")

# 2) GX → 正規化して突合
gx = pd.read_csv(GX_CSV, header=None, dtype=str)
gx_names = gx.iloc[:,0].dropna().astype(str)
gx_df = pd.DataFrame({"gx_company_name": gx_names})
gx_df["gx_company_name_norm"] = gx_df["gx_company_name"].map(norm)

merged = gx_df.merge(df_map[["company_name","ticker","company_name_norm"]],
                     left_on="gx_company_name_norm",
                     right_on="company_name_norm",
                     how="left")

matched = merged.dropna(subset=["ticker"])[["gx_company_name","ticker","company_name"]].rename(columns={"company_name":"mapped_company_name"})
unmatched = merged[merged["ticker"].isna()][["gx_company_name"]]

matched.to_csv(out_proc / "gxleague_tickers.csv", index=False, encoding="utf-8-sig")
unmatched.to_csv(out_proc / "gxleague_unmatched_companies.csv", index=False, encoding="utf-8-sig")

print("GX total:", len(gx_df))
print("Matched :", len(matched))
print("Unmatched:", len(unmatched))
print("Example yfinance ticker:", matched["ticker"].head(5).map(lambda x: f"{x}.T").tolist())