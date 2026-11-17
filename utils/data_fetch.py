"""Live data helpers (CoinGecko, no API key). Every call degrades to cached data, never raises."""
from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import requests
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "btc-econometrics-lab/2.1 (educational app)", "Accept": "application/json"}
TIMEOUT = 8


def _cached_csv(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{ticker}.csv", index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.astype(float)


def _fallback_prices() -> dict:
    out = {"source": "offline", "fetched_at": dt.datetime.now().strftime("%H:%M:%S")}
    for key, tk in (("btc", "BTC_USD"), ("eth", "ETH_USD")):
        c = _cached_csv(tk)["Close"]
        out[f"{key}_price"] = float(c.iloc[-1])
        out[f"{key}_24h"] = float(100 * (c.iloc[-1] / c.iloc[-2] - 1))
    out["note"] = f"OFFLINE FALLBACK - cached prices from {_cached_csv('BTC_USD').index[-1].date()}, not live."
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_prices() -> dict:
    """BTC and ETH spot (USD) with 24h change. Refreshed at most once a minute."""
    try:
        r = requests.get(f"{BASE}/simple/price", headers=HEADERS, timeout=TIMEOUT,
                         params={"ids": "bitcoin,ethereum", "vs_currencies": "usd", "include_24hr_change": "true"})
        r.raise_for_status()
        j = r.json()
        return {"source": "live", "fetched_at": dt.datetime.now().strftime("%H:%M:%S"),
                "btc_price": float(j["bitcoin"]["usd"]), "btc_24h": float(j["bitcoin"]["usd_24h_change"]),
                "eth_price": float(j["ethereum"]["usd"]), "eth_24h": float(j["ethereum"]["usd_24h_change"]),
                "note": "Live from CoinGecko (public API, may be delayed)."}
    except Exception:
        return _fallback_prices()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(days: int = 120) -> tuple[pd.Series, dict]:
    """Daily BTC closes (USD). The last point is today's partial day when live."""
    try:
        r = requests.get(f"{BASE}/coins/bitcoin/market_chart", headers=HEADERS, timeout=TIMEOUT,
                         params={"vs_currency": "usd", "days": days, "interval": "daily"})
        r.raise_for_status()
        p = r.json()["prices"]
        s = pd.Series([x[1] for x in p], index=pd.to_datetime([x[0] for x in p], unit="ms").normalize(), name="Close")
        s = s[~s.index.duplicated(keep="last")].astype(float)
        if len(s) < 40:
            raise ValueError("too few points")
        return s, {"source": "live", "note": f"Live daily closes from CoinGecko, {s.index[0].date()} to {s.index[-1].date()}."}
    except Exception:
        c = _cached_csv("BTC_USD")["Close"].tail(days)
        c.index = pd.to_datetime(c.index).normalize()
        return c, {"source": "offline",
                   "note": f"OFFLINE FALLBACK - cached history ending {c.index[-1].date()} (live API unreachable)."}
