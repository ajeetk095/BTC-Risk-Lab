"""Robust data access layer.

Design goal: the app must never crash for a visitor. Order of preference
  1. committed CSV cache  (always available, reproducible, no network)
  2. live Yahoo Finance download (only if the user explicitly asks)
  3. deterministic synthetic fallback (last resort, clearly labelled)
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DEFAULT_START, DEFAULT_END = "2020-01-01", "2025-02-28"
OUTLIER_MULT = 5.0
CRYPTO = {"BTC-USD", "ETH-USD"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and keep a stable OHLCV schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df]
    df = df[keep].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def load_cached(ticker: str = "BTC-USD") -> pd.DataFrame:
    path = DATA / f"{ticker.replace('-', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cache miss: {path}. Run scripts/refresh_data.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _flat(df)


def load_live(ticker: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, group_by="column")
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {ticker}")
    return _flat(df)


def synthetic(ticker: str = "BTC-USD", n: int = 1885, seed: int = 7) -> pd.DataFrame:
    """Deterministic regime-switching price path. NOT real data - demo only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(DEFAULT_START, periods=n, freq="D")
    vol = np.full(n, 0.02)
    state = 0
    for t in range(1, n):
        p_stay = 0.97 if state == 0 else 0.90
        state = state if rng.random() < p_stay else 1 - state
        vol[t] = 0.018 if state == 0 else 0.055
    r = rng.standard_t(4, n) * vol / np.sqrt(2)
    close = 7200 * np.exp(np.cumsum(r))
    volume = np.exp(rng.normal(24.0, 0.45, n)) * (1 + 6 * np.abs(r))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": volume}, index=idx
    )


def get_data(ticker: str = "BTC-USD", source: str = "cache",
             start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Return (frame, provenance_string)."""
    if source == "live":
        try:
            df = load_live(ticker, start, end)
            return df, f"Live Yahoo Finance - {ticker} ({start} to {end})"
        except Exception as exc:
            try:
                df = load_cached(ticker)
                return df, f"Cached CSV (live download failed: {type(exc).__name__})"
            except Exception:
                return synthetic(ticker), "SYNTHETIC fallback - not real market data"
    try:
        df = load_cached(ticker)
        return df, f"Cached Yahoo Finance CSV - {ticker}"
    except Exception:
        return synthetic(ticker), "SYNTHETIC fallback - not real market data"


# --------------------------------------------------------------------------- #
# transformation - the single source of truth for every model
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame, ticker: str = "BTC-USD") -> pd.DataFrame:
    """Build the modelling frame.

    Key fixes vs. the original notebook
    ----------------------------------
    * flat columns (no tuple column labels leaking into regression tables)
    * explicit daily frequency -> kills statsmodels 'No frequency information'
    * volume in billions and log-volume -> comparable scale to price, so VAR
      coefficients are readable and condition numbers are sane
    * returns in PERCENT -> fixes the arch DataScaleWarning and the
      mis-converged GARCH parameters
    * no in-place mutation of a shared frame
    * single-day volume artefacts are detected and interpolated (see below)
    """
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"].astype(float)
    out["Volume"] = df["Volume"].astype(float)

    # crypto trades 7 days/week; equities do not. Only force 'D' for crypto.
    if ticker in CRYPTO:
        out = out.asfreq("D")
    out = out.dropna()

    # --- volume outlier guard -------------------------------------------------
    # BTC-USD shows a single ~3.5e11 print on 2021-02-26 (11x the local median)
    # that is 5.8x its local median and reverts to normal the next day (a data-error
    # signature). It created a -110k residual in the Engle-Granger regression and
    # a +/-3e11 spike in the differenced volume. Any day whose volume exceeds 5x
    # its centred 61-day rolling median is replaced by linear interpolation (the
    # next-largest ratio is 4.0x - the genuine Nov-2022 FTX days are kept); the raw series is kept in `Volume_raw` and flagged.
    raw_vol = out["Volume"].copy()
    local_med = raw_vol.rolling(61, center=True, min_periods=15).median()
    bad = raw_vol > OUTLIER_MULT * local_med
    out["Volume_raw"] = raw_vol
    out["vol_outlier"] = bad
    out["Volume"] = raw_vol.mask(bad).interpolate(limit_direction="both")

    out["lnClose"] = np.log(out["Close"])
    out["Vol_bn"] = out["Volume"] / 1e9
    out["lnVolume"] = np.log(out["Volume"].replace(0, np.nan))
    out["ret_pct"] = 100.0 * out["Close"].pct_change()
    out["dClose"] = out["Close"].diff()
    out["dVol_bn"] = out["Vol_bn"].diff()
    out["dlnClose"] = out["lnClose"].diff()
    out["dlnVolume"] = out["lnVolume"].diff()
    return out.dropna()


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def quiet():
    """Silence the benign, expected warnings so output stays readable."""
    for cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=cat)
    try:
        from statsmodels.tools.sm_exceptions import (
            InterpolationWarning, ValueWarning, ConvergenceWarning,
        )
        for cat in (InterpolationWarning, ValueWarning, ConvergenceWarning):
            warnings.filterwarnings("ignore", category=cat)
    except Exception:
        pass
