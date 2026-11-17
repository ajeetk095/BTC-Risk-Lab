"""Download and cache the datasets used by the project.

Run once locally:  python scripts/refresh_data.py
The CSVs are committed to the repo so the deployed app works even when
Yahoo Finance blocks the cloud host's IP (a very common deployment failure).
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

START, END = "2020-01-01", "2025-02-28"
TICKERS = ["BTC-USD", "ETH-USD", "GOOGL", "GOOG"]


def clean(ticker: str) -> pd.DataFrame:
    """Download one ticker and return a FLAT-column OHLCV frame.

    Fixes the two silent yfinance breakages:
      * MultiIndex ('Price','Ticker') columns  -> flattened
      * auto_adjust default flipped to True    -> pinned to False
    """
    df = yf.download(ticker, start=START, end=END, progress=False,
                     auto_adjust=False, group_by="column")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def main() -> int:
    for t in TICKERS:
        try:
            df = clean(t)
            path = DATA / f"{t.replace('-', '_')}.csv"
            df.to_csv(path)
            print(f"OK  {t:9s} {len(df):5d} rows -> {path.name}")
        except Exception as exc:                     # keep going, report at end
            print(f"FAIL {t}: {type(exc).__name__}: {exc}", file=sys.stderr)
    meta = pd.DataFrame({"key": ["start", "end", "source", "auto_adjust"],
                         "value": [START, END, "Yahoo Finance via yfinance", "False"]})
    meta.to_csv(DATA / "metadata.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
