# Paths to Update
## notebooks/import requests.py
- L181:     out_dir = Path("data/data_collection/processed/eegs_debug")

## scripts/2nd screening/explanation_env_emission.py
- L113: _generated_envlink_YYYYMMDD_HHMMSS/

## scripts/2nd screening/fetch_env_emission.py
- L311:     return out_root / f"_generated_envlink_{ts}"
- L440:     # outputs

## scripts/2nd screening/fetch_env_graph_exact.py
- L9:   - _generated_envgraph_YYYYMMDD_HHMMSS/ (raw json, long csv, logs, summary)
- L248:     outdir = os.path.join(out_root, f"_generated_envgraph_{now_tag()}")
- L380:     # final outputs

## scripts/2nd screening/fetch_env_make_array.py
- L6:   .../_generated_envgraph_YYYYMMDD_HHMMSS/raw_json/*.json
- L201:     # Write outputs

## scripts/2nd screening/toplank_energyCO2_analize.py
- L109:     # Save outputs

## scripts/build_climate_commitment_flags.py
- L8: - src/data_collection/processed/universe_with_gx_flag.csv
- L10: - src/data_collection/processed/jpx_company_ticker_map.csv
- L12: - src/data_collection/processed/jpx_ticker_english_name.csv
- L14: - src/data_collection/raw/TCFDcompanies.csv
- L16: - src/data_collection/raw/SBTis_Target_Dashboard.csv
- L20: - src/data_collection/processed/climate_commitment_flags.csv
- L22: - src/data_collection/processed/climate_commitment_flags_unmatched.csv
- L534:         default="src/data_collection/processed/universe_with_gx_flag.csv",
- L539:         default="src/data_collection/processed/jpx_company_ticker_map.csv",
- L544:         default="src/data_collection/processed/jpx_ticker_english_name.csv",
- L549:         default="src/data_collection/processed/tcfd_company_names.txt",
- L554:         default="src/data_collection/raw/SBTis_Target_Dashboard.csv",
- L559:         default="src/data_collection/processed/climate_commitment_flags.csv",
- L564:         default="src/data_collection/processed/climate_commitment_flags_unmatched.csv",
- L569:         default="src/data_collection/processed/universe_operating_companies.csv",
- L574:         default="src/data_collection/processed/climate_commitment_scores.csv",

## scripts/build_gx_rank_ticker_alignment.py
- L6: RANK_PATH = BASE / "data/data_collection/processed/GXleague_rank.csv"
- L7: TICK_PATH = BASE / "data/data_collection/processed/gxleague_tickers.csv"
- L9: OUT_PATH = BASE / "data/data_collection/processed/gxleague_tickers_with_rank_1_2.csv"
- L10: UNMATCHED_PATH = BASE / "data/data_collection/processed/gxleague_tickers_unmatched_in_rank.csv"
- L11: DUP_RANK_PATH = BASE / "data/data_collection/processed/GXleague_rank_duplicates_by_normname.csv"

## scripts/build_jpx_english_name_map.py
- L8: - src/data_collection/processed/universe_with_gx_flag.csv
- L12: - src/data_collection/processed/jpx_ticker_english_name.csv
- L15: - src/data_collection/processed/universe_operating_companies.csv
- L277:         default="src/data_collection/processed/universe_with_gx_flag.csv",
- L282:         default="src/data_collection/processed/jpx_ticker_english_name.csv",
- L287:         default="src/data_collection/processed/universe_operating_companies.csv",

## scripts/build_ticker_map_from_jpx.py
- L5: JPX_XLS = "/Users/shou/hobby/CPX/nikkei-stock/src/data_collection/data_j.xls"   # ←あなたのファイル名に合わせる
- L6: GX_CSV  = "/Users/shou/hobby/CPX/nikkei-stock/src/data_collection/GXleague_names.csv"
- L8: out_raw = Path("data/raw"); out_raw.mkdir(parents=True, exist_ok=True)
- L9: out_proc = Path("data/processed"); out_proc.mkdir(parents=True, exist_ok=True)

## scripts/build_universe_with_gx_flag.py
- L4: # Data base directory (you said you manage data under src/data_collection)
- L5: BASE = Path("src/data_collection")
- L14:     Path("data/raw/jpx_company_ticker_map.csv"),
- L15:     Path("data/processed/jpx_company_ticker_map.csv"),
- L21:     Path("data/processed/gxleague_tickers.csv"),

## scripts/eegs_scrape_batch.py
- L6: search_result2023.csv のURL列を順番に処理して
- L7: 「会社ごとに1CSV」を out/eegs/ に出力するバッチスクリプト。
- L10: - 入力: data/search_result2023.csv
- L13: - 出力: out/eegs/companies/<会社名>_<特定排出者コード>.csv
- L15:   - out/eegs/errors.csv (失敗URL)
- L16:   - out/eegs/manifest.csv (成功した会社ファイル一覧)
- L17:   - out/eegs/eegs_master.csv (全社縦結合のマスター; 省略したければOFF可能)
- L40: INPUT_CSV = "data/data_collection/raw/search_result_2023.csv"
- L46: OUT_ROOT = "out/eegs"
- L63:     "data/search_result2023.csv",
- L64:     "data/search_result2023_utf8.csv",
- L65:     "data/search_result2023_sjis.csv",
- L66:     "data/search_result2023_2023.csv",
- L67:     "data/search_result2023.csv",
- L68:     "data/raw/search_result2023.csv",
- L69:     "data/raw/search_result2023_2023.csv",
- L70:     "data/data_collection/raw/search_result_2023.csv",
- L71:     "data/data_collection/raw/search_result_2023_utf8.csv",
- L72:     "search_result2023.csv",
- L95:     for pat in ["data/**/*search_result*2023*.csv", "data/**/*search_result*.csv"]:
- L103:         "入力CSVが見つかりません。次を確認してください: data/search_result2023.csv（または --input で指定）。\n"

## scripts/fetch_yfinance_gx_financials.py
- L14: IN_PATH = BASE / "data/data_collection/processed/gxleague_tickers_with_rank_1_2.csv"
- L18: OUT_DIR = BASE / f"data/data_collection/processed/yfinance_gx_{RUN_TS}"

