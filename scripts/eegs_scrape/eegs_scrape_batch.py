#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEGS（環境省 温室効果ガス排出量算定・報告・公表制度）サイトから、
search_result2023.csv のURL列を順番に処理して
「会社ごとに1CSV」を out/eegs/ に出力するバッチスクリプト。

要件:
- 入力: data/search_result2023.csv
  - 右端のURL列（または列名指定）から corporate URL を読む
- 処理: corporate -> 特定事業所一覧 -> 各事業所詳細 の巡回
- 出力: out/eegs/companies/<会社名>_<特定排出者コード>.csv
- 併せてログ:
  - out/eegs/errors.csv (失敗URL)
  - out/eegs/manifest.csv (成功した会社ファイル一覧)
  - out/eegs/eegs_master.csv (全社縦結合のマスター; 省略したければOFF可能)

依存:
pip install requests beautifulsoup4 pandas lxml
"""

import os
import sys
from pathlib import Path
import re
import time
import random
import traceback
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------
# 設定
# ----------------------------
INPUT_CSV = "data/data_collection/raw/search_result_2023.csv"

# URL列名が分かるならここに固定（例: "ページ"）。
# 分からない/右端を使うなら None のままにする。
URL_COLUMN_NAME = None  # 例: "ページ"

OUT_ROOT = "out/eegs"
OUT_COMPANY_DIR = os.path.join(OUT_ROOT, "companies")

WRITE_MASTER = True  # 全社縦結合の master を出したくなければ False

SLEEP_SEC_BASE = 0.3
SLEEP_JITTER = 0.2  # 0〜この値を足してランダム待機

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; +https://example.com/bot)"
}

TIMEOUT_SEC = 30
MAX_RETRIES = 3

# 入力CSVの候補（固定パスが無い／違う場合に自動探索する）
INPUT_CSV_CANDIDATES = [
    "data/search_result2023.csv",
    "data/search_result2023_utf8.csv",
    "data/search_result2023_sjis.csv",
    "data/search_result2023_2023.csv",
    "data/search_result2023.csv",
    "data/raw/search_result2023.csv",
    "data/raw/search_result2023_2023.csv",
    "data/data_collection/raw/search_result_2023.csv",
    "data/data_collection/raw/search_result_2023_utf8.csv",
    "search_result2023.csv",
]


def resolve_input_csv(path_str: str | None) -> str:
    """入力CSVパスを解決する。

    - 指定パスが存在すればそれを使う
    - 存在しない場合は候補・data配下を探索
    """
    # 1) 明示指定
    if path_str:
        p = Path(path_str)
        if p.exists():
            return str(p)

    # 2) 既定候補
    for cand in INPUT_CSV_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return str(p)

    # 3) data配下をゆるく探索（最初に見つかったものを採用）
    for pat in ["data/**/*search_result*2023*.csv", "data/**/*search_result*.csv"]:
        found = sorted(Path(".").glob(pat))
        if found:
            return str(found[0])

    tried = [path_str] if path_str else []
    tried += INPUT_CSV_CANDIDATES
    raise FileNotFoundError(
        "入力CSVが見つかりません。次を確認してください: data/search_result2023.csv（または --input で指定）。\n"
        + "Tried:\n  - " + "\n  - ".join([t for t in tried if t])
    )


def parse_cli_args(argv: list[str]):
    """超簡易CLI: 依存を増やさずに --input と --url-col だけ受ける。"""
    input_path = None
    url_col = None
    for i, a in enumerate(argv):
        if a == "--input" and i + 1 < len(argv):
            input_path = argv[i + 1]
        if a == "--url-col" and i + 1 < len(argv):
            url_col = argv[i + 1]
    return input_path, url_col


# ----------------------------
# ユーティリティ
# ----------------------------
def ensure_dirs():
    os.makedirs(OUT_COMPANY_DIR, exist_ok=True)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_filename(s: str) -> str:
    """macOS/Windows両対応で壊れにくいファイル名に寄せる

    - / \\ : * ? " < > | などを置換
    - 連続スペースを1つに
    """
    s = normalize_ws(s)
    s = re.sub(r'[\/\\:\*\?"<>\|]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "unknown_company"
    return s[:120]  # 長すぎ防止


def to_int_from_emission_text(text: str):
    if text is None:
        return None
    t = text.replace(",", "")
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else None


def extract_sp_emit_code(url: str):
    """
    corporate URL 例:
    https://eegs.env.go.jp/ghg-santeikohyo-result/corporate?spEmitCode=985452408&repDivID=1
    """
    try:
        q = parse_qs(urlparse(url).query)
        return (q.get("spEmitCode") or [None])[0]
    except Exception:
        return None


def polite_sleep():
    time.sleep(SLEEP_SEC_BASE + random.random() * SLEEP_JITTER)


# ----------------------------
# HTTP取得
# ----------------------------
def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            last_err = e
            # リトライ前に少し待つ（指数バックオフ）
            time.sleep(0.8 * attempt)
    raise last_err


# ----------------------------
# ラベル抽出（壊れにくい方針）
# ----------------------------
def extract_value_by_label(soup: BeautifulSoup, label: str):
    label_node = soup.find(string=lambda x: x and x.strip() == label)
    if not label_node:
        return None

    # ラベルの親以降から最初の非空文字を拾う
    for cand in label_node.parent.find_all_next(string=True, limit=120):
        t = cand.strip()
        if t and t != label:
            return t
    return None


def parse_corporate_info(corp_soup: BeautifulSoup):
    keys = [
        "事業者名",
        "法人番号",
        "特定排出者コード",
        "所在地",
        "主たる事業",
        "従業員数",
        "株式銘柄コード(任意入力)",
        "ISINコード(任意入力)",
    ]
    info = {k: extract_value_by_label(corp_soup, k) for k in keys}
    return {
        "事業者名": info.get("事業者名"),
        "法人番号": info.get("法人番号"),
        "特定排出者コード": info.get("特定排出者コード"),
        "所在地": info.get("所在地"),
        "主たる事業": info.get("主たる事業"),
        "従業員数": info.get("従業員数"),
        "株式銘柄コード": info.get("株式銘柄コード(任意入力)"),
        "ISINコード": info.get("ISINコード(任意入力)"),
    }


def parse_office_list_from_corporate(corp_soup: BeautifulSoup, base_url: str):
    table = corp_soup.find("table")
    if not table:
        raise RuntimeError("事業所一覧テーブルが見つかりませんでした。DOMが変わった可能性があります。")

    offices = []
    trs = table.find_all("tr")
    for tr in trs[1:]:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        office_name = tds[0].get_text(strip=True)
        business = tds[1].get_text(strip=True)
        location = tds[2].get_text(strip=True)
        emission_text = tds[3].get_text(strip=True)
        emission_num = to_int_from_emission_text(emission_text)

        link = tds[4].find("a")
        detail_url = urljoin(base_url, link["href"]) if link and link.has_attr("href") else None

        offices.append({
            "事業所名": office_name,
            "一覧_事業": business,
            "一覧_所在地": location,
            "一覧_温室効果ガス算定排出量_text": emission_text,
            "一覧_温室効果ガス算定排出量_tCO2": emission_num,
            "詳細ページURL": detail_url,
        })

    return offices


def parse_office_detail(office_soup: BeautifulSoup):
    detail = {
        "詳細_事業者名": extract_value_by_label(office_soup, "事業者名"),
        "詳細_事業所名": extract_value_by_label(office_soup, "事業所名"),
        "詳細_所在都道府県": extract_value_by_label(office_soup, "所在都道府県"),
        "詳細_特定排出者コード": extract_value_by_label(office_soup, "特定排出者コード"),
        "詳細_事業所において行われる事業": extract_value_by_label(office_soup, "事業所において行われる事業"),
    }

    # 推移（合計）: 年(20xx)の直前の整数を拾う簡易法
    text = normalize_ws(office_soup.get_text(" ", strip=True))
    tokens = re.findall(r"\b\d+\b", text)

    trend = {}
    prev = None
    for tok in tokens:
        if re.fullmatch(r"20\d{2}", tok):
            if prev and not re.fullmatch(r"20\d{2}", prev):
                trend[f"推移_合計_千tCO2_{tok}"] = int(prev)
        prev = tok

    detail.update(trend)
    return detail


# ----------------------------
# 企業1社ぶんをスクレイプ -> DataFrame
# ----------------------------
def scrape_one_company(session: requests.Session, corporate_url: str) -> pd.DataFrame:
    corp_soup = fetch_soup(session, corporate_url)
    corp_info = parse_corporate_info(corp_soup)
    offices = parse_office_list_from_corporate(corp_soup, base_url=corporate_url)

    rows = []
    for off in offices:
        polite_sleep()
        if not off.get("詳細ページURL"):
            rows.append({**corp_info, **off})
            continue

        office_soup = fetch_soup(session, off["詳細ページURL"])
        detail = parse_office_detail(office_soup)
        rows.append({**corp_info, **off, **detail})

    return pd.DataFrame(rows)


# ----------------------------
# 入力CSVからURL列を取得
# ----------------------------
def load_urls_from_input(csv_path: str, url_col_name: str | None = None):
    # 文字コードが不明な場合があるため、utf-8-sig → utf-8 → cp932 の順に試す
    read_err = None
    for enc in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            read_err = e
            df = None
    if df is None:
        raise read_err

    if df.empty:
        return [], "(empty)"

    if url_col_name and url_col_name in df.columns:
        col = url_col_name
    elif URL_COLUMN_NAME and URL_COLUMN_NAME in df.columns:
        col = URL_COLUMN_NAME
    else:
        # 右端列をURL列として扱う
        col = df.columns[-1]

    urls = df[col].dropna().astype(str).tolist()
    # それっぽいURLだけ残す（念のため）
    urls = [u for u in urls if u.startswith("http")]
    return urls, col


# ----------------------------
# メイン
# ----------------------------
def main():
    ensure_dirs()

    cli_input, cli_url_col = parse_cli_args(sys.argv[1:])
    input_path = resolve_input_csv(cli_input or INPUT_CSV)

    urls, used_col = load_urls_from_input(input_path, url_col_name=cli_url_col)
    print(f"[INFO] input={input_path}  url_column={used_col}  urls={len(urls)}")

    session = requests.Session()

    manifest_rows = []
    error_rows = []
    master_dfs = []

    for idx, url in enumerate(urls, start=1):
        polite_sleep()
        sp_code = extract_sp_emit_code(url)

        try:
            print(f"[{idx}/{len(urls)}] scraping: spEmitCode={sp_code} url={url}")

            df_company = scrape_one_company(session, url)
            if df_company.empty:
                raise RuntimeError("企業DataFrameが空でした（事業所が0件 or 解析失敗の可能性）")

            # ファイル名：会社名 + 特定排出者コード（URL由来を優先）
            corp_name = df_company.get("事業者名")
            corp_name = corp_name.iloc[0] if corp_name is not None and len(corp_name) > 0 else None
            corp_name = corp_name or df_company.get("詳細_事業者名", pd.Series([None])).iloc[0] or "unknown_company"

            code_for_name = sp_code or df_company.get("特定排出者コード", pd.Series([None])).iloc[0] or "unknown_code"

            filename = f"{safe_filename(corp_name)}_{safe_filename(str(code_for_name))}.csv"
            out_path = os.path.join(OUT_COMPANY_DIR, filename)

            df_company.to_csv(out_path, index=False, encoding="utf-8-sig")

            manifest_rows.append({
                "index": idx,
                "事業者名": corp_name,
                "spEmitCode": sp_code,
                "特定排出者コード": df_company.get("特定排出者コード", pd.Series([None])).iloc[0],
                "corporate_url": url,
                "rows": len(df_company),
                "cols": len(df_company.columns),
                "output_path": out_path,
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
            })

            if WRITE_MASTER:
                # masterに入れる用に会社識別列を付与
                df_company2 = df_company.copy()
                df_company2.insert(0, "会社CSV", filename)
                df_company2.insert(1, "corporate_url", url)
                master_dfs.append(df_company2)

        except Exception as e:
            err_msg = str(e)
            error_rows.append({
                "index": idx,
                "spEmitCode": sp_code,
                "corporate_url": url,
                "error": err_msg,
                "traceback": traceback.format_exc(limit=5),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[ERROR] {idx} url={url} err={err_msg}")

    # manifest / errors
    manifest_path = os.path.join(OUT_ROOT, "manifest.csv")
    errors_path = os.path.join(OUT_ROOT, "errors.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(errors_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Saved manifest: {manifest_path} rows={len(manifest_rows)}")
    print(f"[INFO] Saved errors  : {errors_path} rows={len(error_rows)}")

    # master
    if WRITE_MASTER and master_dfs:
        master = pd.concat(master_dfs, axis=0, ignore_index=True)
        master_path = os.path.join(OUT_ROOT, "eegs_master.csv")
        master.to_csv(master_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved master  : {master_path} rows={len(master)} cols={len(master.columns)}")

    print("[DONE]")


if __name__ == "__main__":
    main()