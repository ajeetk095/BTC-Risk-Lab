"""Challenge 1 - Non-stationarity and finding an equilibrium.

ADF / KPSS -> integration order -> Engle-Granger cointegration ->
ECM (only when cointegration is supported) -> VAR on differenced data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, kpss

from .data import quiet

quiet()


# --------------------------------------------------------------------------- #
# unit roots
# --------------------------------------------------------------------------- #
def unit_root_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) side by side.

    KPSS p-values are truncated to [0.01, 0.10] by statsmodels; we report the
    bound explicitly instead of pretending '0.1' is an exact p-value.
    """
    rows = []
    for c in cols:
        s = df[c].dropna()
        a = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(s, regression="c", nlags="auto")
        kp = float(k[1])
        kp_txt = ">= 0.10" if kp >= 0.10 else ("<= 0.01" if kp <= 0.01 else f"{kp:.4f}")
        adf_reject = a[1] < 0.05
        kpss_reject = kp < 0.05
        if adf_reject and not kpss_reject:
            verdict = "I(0) stationary"
        elif not adf_reject and kpss_reject:
            verdict = "I(1) unit root"
        elif adf_reject and kpss_reject:
            verdict = "conflicting - possible fractionally integrated / structural break"
        else:
            verdict = "inconclusive - both fail to reject"
        rows.append(dict(series=c, n=int(s.size), ADF_stat=float(a[0]),
                         ADF_p=float(a[1]), ADF_lags=int(a[2]),
                         KPSS_stat=float(k[0]), KPSS_p=kp_txt, verdict=verdict))
    return pd.DataFrame(rows)


def integration_order(df: pd.DataFrame, col: str) -> dict:
    """Test level and first difference so cointegration preconditions are explicit."""
    lvl = adfuller(df[col].dropna(), autolag="AIC")
    dif = adfuller(df[col].dropna().diff().dropna(), autolag="AIC")
    level_stat = lvl[1] >= 0.05
    diff_stat = dif[1] < 0.05
    order = 1 if (level_stat and diff_stat) else (0 if not level_stat else 2)
    return dict(series=col, ADF_level_p=float(lvl[1]), ADF_diff_p=float(dif[1]),
                order=order,
                note={0: "already stationary in levels",
                      1: "I(1): stationary after first differencing",
                      2: "needs further investigation"}[order])


# --------------------------------------------------------------------------- #
# Engle-Granger
# --------------------------------------------------------------------------- #
def engle_granger(df: pd.DataFrame, y: str, x: str, log: bool = False) -> dict:
    """Two-step Engle-Granger with the CORRECT critical values.

    The original code ran a plain ADF on the OLS residuals and read off the
    ordinary Dickey-Fuller p-value. Residual-based tests need MacKinnon's
    cointegration critical values, which statsmodels.tsa.stattools.coint supplies.
    """
    Y, X = (np.log(df[y]), np.log(df[x])) if log else (df[y], df[x])
    keep = pd.concat([Y, X], axis=1).dropna()
    Y, X = keep.iloc[:, 0], keep.iloc[:, 1]

    orders = [integration_order(df, y), integration_order(df, x)]
    precondition = all(o["order"] == 1 for o in orders)

    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    naive = adfuller(ols.resid, autolag="AIC")
    stat, pval, crit = coint(Y, X, trend="c", autolag="AIC")

    return dict(y=y, x=x, log=log, n=int(keep.shape[0]), ols=ols,
                beta=float(ols.params.iloc[1]), alpha=float(ols.params.iloc[0]),
                r2=float(ols.rsquared), dw=float(sm.stats.durbin_watson(ols.resid)),
                resid=pd.Series(ols.resid.values, index=keep.index, name="ec_gap"),
                naive_adf_p=float(naive[1]), eg_stat=float(stat), eg_p=float(pval),
                eg_crit={k: float(v) for k, v in zip(["1%", "5%", "10%"], crit)},
                cointegrated=bool(pval < 0.05),
                orders=orders, precondition_ok=bool(precondition),
                condition_number=float(ols.condition_number))


def fit_ecm(eg: dict, df: pd.DataFrame, y: str, x: str, log: bool = False) -> dict:
    """Error-correction model. Only meaningful when the pair is cointegrated."""
    Y, X = (np.log(df[y]), np.log(df[x])) if log else (df[y], df[x])
    d = pd.DataFrame({"dY": Y.diff(), "dX": X.diff()})
    d["ec_lag"] = eg["resid"].shift(1)
    d = d.dropna()
    res = sm.OLS(d["dY"], sm.add_constant(d[["ec_lag", "dX"]])).fit()
    phi = float(res.params["ec_lag"])
    p = float(res.pvalues["ec_lag"])
    half_life = float(np.log(0.5) / np.log(1 + phi)) if -2 < phi < 0 else np.nan
    return dict(model=res, n=int(d.shape[0]), phi=phi, phi_p=p,
                gamma=float(res.params["dX"]), r2=float(res.rsquared),
                half_life_days=half_life, data=d,
                valid=bool(eg["cointegrated"] and phi < 0 and p < 0.05),
                speed_note=("disequilibrium is corrected" if phi < 0 and p < 0.05
                            else "no statistically significant error correction"))


# --------------------------------------------------------------------------- #
# VAR
# --------------------------------------------------------------------------- #
def fit_var(df: pd.DataFrame, cols, criterion: str = "bic", maxlags: int | None = None):
    """VAR on stationary, comparably-scaled variables.

    Fixes: data-driven lag cap (Schwert rule) instead of a hard-coded 15 that
    the AIC was pinned against; scale-matched inputs; BIC/HQIC parsimony;
    plus the stability / whiteness diagnostics the original never ran.
    """
    data = df[list(cols)].dropna()
    n = len(data)
    if maxlags is None:
        maxlags = int(min(20, max(1, 12 * (n / 100) ** 0.25)))
    model = VAR(data)
    sel = model.select_order(maxlags)
    orders = {k: int(v) for k, v in sel.selected_orders.items()}
    p = max(1, orders.get(criterion, 1))
    res = model.fit(p)

    stable = bool(res.is_stable())
    try:
        lb = res.test_whiteness(nlags=min(20, max(p + 1, 10)), adjusted=True)
        white_p = float(lb.pvalue)
    except Exception:
        white_p = np.nan
    try:
        gc = res.test_causality(caused=cols[0], causing=[cols[1]], kind="f")
        granger = dict(test=f"{cols[1]} -> {cols[0]}", stat=float(gc.test_statistic),
                       p=float(gc.pvalue))
    except Exception:
        granger = None
    return dict(result=res, data=data, maxlags=maxlags, orders=orders,
                chosen_criterion=criterion, p=p, n=n, stable=stable,
                whiteness_p=white_p, granger=granger,
                aic=float(res.aic), bic=float(res.bic))
