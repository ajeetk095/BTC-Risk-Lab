"""Challenge 4 - Feature extraction.

Conditional mean (ARMA), conditional variance (GARCH family), squared returns
and a Kalman-filtered local level. All estimated on PERCENT returns so the
optimiser is well scaled, with an explicit persistence / stationarity check.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .data import quiet

quiet()

SPECS = {
    "GARCH(1,1)-t": dict(vol="GARCH", p=1, q=1, dist="t"),
    "GARCH(1,1)-normal": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "GJR-GARCH(1,1)-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH(1,1)-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}


def arma_mean(returns_pct: pd.Series, order=(1, 0, 1)) -> dict:
    s = returns_pct.dropna().astype(float)
    res = ARIMA(s, order=order).fit()
    return dict(result=res, order=order,
                conditional_mean=pd.Series(res.fittedvalues, index=s.index,
                                           name="conditional_mean"),
                resid=pd.Series(res.resid, index=s.index, name="arma_resid"),
                aic=float(res.aic), bic=float(res.bic))


def garch_variance(returns_pct: pd.Series, spec: str = "GARCH(1,1)-t",
                   mean: str = "Constant") -> dict:
    """Fit a volatility model on percent returns with rescale=False.

    Fixes the original DataScaleWarning (scale 0.0011) that produced
    alpha=0.67 / beta=0.04 and the wrong conclusion 'no volatility clustering'.
    """
    s = returns_pct.dropna().astype(float)
    if float(s.std()) < 0.05:
        raise ValueError("Returns look unscaled - pass PERCENT returns (100 * pct_change).")
    am = arch_model(s, mean=mean, rescale=False, **SPECS[spec])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)

    p = res.params
    alpha = float(p.get("alpha[1]", np.nan))
    beta = float(p.get("beta[1]", np.nan))
    gamma = float(p.get("gamma[1]", 0.0))
    if spec.startswith("EGARCH"):
        persistence = beta
        omega, unconditional = np.nan, np.nan
        half_life = np.log(0.5) / np.log(beta) if 0 < beta < 1 else np.inf
    else:
        persistence = alpha + beta + (gamma / 2.0 if "GJR" in spec else 0.0)
        omega = float(p.get("omega", np.nan))
        unconditional = (np.sqrt(omega / (1 - persistence))
                         if persistence < 1 and omega > 0 else np.nan)
        half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf

    cond_vol = pd.Series(res.conditional_volatility, index=s.index, name="cond_vol_pct")
    return dict(result=res, spec=spec, params={k: float(v) for k, v in p.items()},
                alpha=alpha, beta=beta, gamma=gamma, omega=omega,
                persistence=float(persistence),
                integrated=bool(persistence >= 0.995),
                # arch constrains alpha+beta <= 1, so a value of exactly 1.0 is a
                # boundary solution of the optimiser, not a free estimate.
                at_boundary=bool((not spec.startswith("EGARCH")) and persistence >= 0.9999),
                unconditional_vol=float(unconditional) if np.isfinite(unconditional) else np.nan,
                half_life_days=float(half_life) if np.isfinite(half_life) else np.inf,
                nu=float(p.get("nu", np.nan)),
                cond_vol=cond_vol, cond_var=cond_vol ** 2,
                loglik=float(res.loglikelihood), aic=float(res.aic), bic=float(res.bic),
                std_resid=pd.Series(res.std_resid, index=s.index, name="std_resid"))


def compare_specs(returns_pct: pd.Series) -> pd.DataFrame:
    rows = []
    for name in SPECS:
        try:
            g = garch_variance(returns_pct, name)
            rows.append(dict(spec=name, loglik=g["loglik"], AIC=g["aic"], BIC=g["bic"],
                             persistence=g["persistence"],
                             nu=g["nu"] if np.isfinite(g["nu"]) else np.nan))
        except Exception as exc:
            rows.append(dict(spec=name, loglik=np.nan, AIC=np.nan, BIC=np.nan,
                             persistence=np.nan, nu=np.nan))
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def kalman_level(returns_pct: pd.Series, level: str = "local level") -> dict:
    s = returns_pct.dropna().astype(float)
    res = UnobservedComponents(s, level=level).fit()
    trend = pd.Series(res.smoothed_state[0], index=s.index, name="kalman_level")
    sig_level = float(res.params.get("sigma2.level", np.nan))
    sig_irr = float(res.params.get("sigma2.irregular", np.nan))
    ratio = sig_level / sig_irr if sig_irr > 0 else np.nan
    return dict(result=res, trend=trend, sigma2_level=sig_level,
                sigma2_irregular=sig_irr, signal_to_noise=ratio,
                aic=float(res.aic),
                interpretation=("level is essentially constant - returns have no "
                                "persistent drift" if ratio < 1e-3 else
                                "time-varying local level"))


def feature_panel(returns_pct: pd.Series, arma_order=(1, 0, 1),
                  spec: str = "GARCH(1,1)-t") -> dict:
    """The four extracted features on one aligned index."""
    a = arma_mean(returns_pct, arma_order)
    g = garch_variance(returns_pct, spec)
    k = kalman_level(returns_pct)
    s = returns_pct.dropna().astype(float)
    panel = pd.DataFrame({
        "return_pct": s,
        "conditional_mean": a["conditional_mean"],
        "conditional_variance": g["cond_var"],
        "squared_return": s ** 2,
        "kalman_level": k["trend"],
    }).dropna()
    corr = panel[["squared_return", "conditional_variance"]].corr().iloc[0, 1]
    return dict(arma=a, garch=g, kalman=k, panel=panel,
                sq_vs_garch_corr=float(corr),
                adf=adfuller(s, autolag="AIC"))


def forecast_risk(garch: dict, horizon: int = 10) -> pd.DataFrame:
    """h-step volatility forecast, plus single-day 95%/99% VaR at each horizon.

    `VaR*_pct` is the VaR of the return on day h (not of the h-day cumulative
    return); `cum_vol_pct` is the volatility of the h-day cumulative return.

    EGARCH has no closed-form multi-step variance forecast in `arch`, so we fall
    back to a simulation average instead of raising 'Analytic forecasts not
    available for horizon > 1'.
    """
    res = garch["result"]
    method = "analytic"
    try:
        f = res.forecast(horizon=horizon, reindex=False)
        var = np.asarray(f.variance.values[-1], dtype=float)
    except Exception:
        method = "simulation"
        f = res.forecast(horizon=horizon, reindex=False,
                         method="simulation", simulations=2000)
        var = np.mean(np.asarray(f.simulations.variances)[-1], axis=0)
    var = np.asarray(var, dtype=float).ravel()[:horizon]
    if var.size < horizon:                      # pad defensively
        var = np.pad(var, (0, horizon - var.size), mode="edge")
    vol = np.sqrt(np.maximum(var, 1e-12))
    nu = garch["nu"]
    if np.isfinite(nu) and nu > 2:
        from scipy.stats import t as student_t
        scale = np.sqrt((nu - 2) / nu)
        # standardised Student-t: z = t * sqrt((nu-2)/nu)  ->  MULTIPLY by scale.
        # (The first version divided, which inflated VaR by 1/scale^2 ~ 2.9x.)
        q95 = student_t.ppf(0.05, nu) * scale
        q99 = student_t.ppf(0.01, nu) * scale
    else:
        from scipy.stats import norm
        q95, q99 = norm.ppf(0.05), norm.ppf(0.01)
    mu = garch["params"].get("mu", 0.0)
    return pd.DataFrame({
        "horizon_days": np.arange(1, horizon + 1),
        "vol_forecast_pct": vol,
        "cum_vol_pct": np.sqrt(np.cumsum(np.maximum(var, 1e-12))),
        "VaR95_pct": mu + q95 * vol,
        "VaR99_pct": mu + q99 * vol,
        "method": method,
    })
