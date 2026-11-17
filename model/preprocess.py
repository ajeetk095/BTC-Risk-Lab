"""Pre-processing shared by training and the live app: prices -> PERCENT returns.

Percent (not decimal) returns are essential: GARCH on decimal returns triggers the
arch DataScaleWarning and converges to the wrong parameters (see Bug lab, item 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_percent_returns(prices: pd.Series) -> pd.Series:
    r = 100.0 * prices.astype(float).pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna().rename("ret_pct")
