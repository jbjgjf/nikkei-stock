

"""
yfinance.py

Yahoo Finance (yfinance) を用いて日本株の株価データを取得し、
CSV として保存するためのモジュール。

このモジュールは「CSVを探す」のではなく、
Pythonコードによって一次スクリーニング用の市場データを生成することを目的とする。
"""

from pathlib import Path
from typing import List
import yfinance as yf
import pandas as pd


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_path: str = "data/processed/price_yfinance.csv",
) -> pd.DataFrame:
    """
    指定した銘柄リストについて、Yahoo Finance から株価データを取得し、
    CSV に保存する。

    Parameters
    ----------
    tickers : List[str]
        例: ["7203.T", "6758.T", "9984.T"]
    start_date : str
        取得開始日（YYYY-MM-DD）
    end_date : str
        取得終了日（YYYY-MM-DD）
    output_path : str
        保存先CSVパス

    Returns
    -------
    pd.DataFrame
        取得した株価データ（MultiIndex列）
    """

    # 出力先ディレクトリを作成
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # yfinance で株価取得
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    # CSV 保存
    df.to_csv(output_file, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    # 動作確認用の簡易テスト
    sample_tickers = ["7203.T", "6758.T", "9984.T"]
    download_price_data(
        tickers=sample_tickers,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )