"""Light-weight inference engine: applies the FITTED model parameters to new data.

No model is re-estimated here - the app only filters/forecasts with parameters that
were estimated offline (scripts/train_model.py -> model/model.pkl). That is what makes
live use fast and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


def garch_filter(ret: np.ndarray, mu: float, omega: float, alpha: float, beta: float, v0: float):
    """Conditional variance path and the one-step-ahead variance."""
    ret = np.asarray(ret, dtype=float)
    var = np.empty(len(ret))
    var[0] = v0
    for i in range(1, len(ret)):
        var[i] = omega + alpha * (ret[i - 1] - mu) ** 2 + beta * var[i - 1]
    nxt = omega + alpha * (ret[-1] - mu) ** 2 + beta * var[-1]
    return var, float(nxt)


def markov_filter(ret: np.ndarray, mu, s2, P) -> np.ndarray:
    """Hamilton filter: P(S_t = k | data up to t), 2+ states, normal densities."""
    ret = np.asarray(ret, dtype=float)
    mu, s2, P = np.asarray(mu, float), np.asarray(s2, float), np.asarray(P, float)
    k = len(mu)
    # stationary distribution of the chain as the initial prior
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = pi / pi.sum()
    out = np.empty((len(ret), k))
    prior = pi
    for i, y in enumerate(ret):
        dens = norm.pdf(y, loc=mu, scale=np.sqrt(s2))
        post = prior * dens
        tot = post.sum()
        post = post / tot if tot > 0 else prior
        out[i] = post
        prior = P.T @ post
    return out


def forecast_variance(next_var: float, omega: float, alpha: float, beta: float, horizon: int) -> np.ndarray:
    """Variance path for days t+1 ... t+h given the one-step variance."""
    v = np.empty(horizon)
    v[0] = next_var
    for k in range(1, horizon):
        v[k] = omega + (alpha + beta) * v[k - 1]
    return v


def quantile_z(level: float, nu: float | None, dist: str = "t") -> float:
    """Lower-tail quantile of a UNIT-VARIANCE innovation (level e.g. 0.05)."""
    if dist == "t" and nu and np.isfinite(nu) and nu > 2:
        return float(student_t.ppf(level, nu) * np.sqrt((nu - 2) / nu))   # multiply, not divide
    return float(norm.ppf(level))


def var_pct(sigma: float, mu: float, nu: float | None, conf: float = 0.95, dist: str = "t") -> float:
    """One-day Value-at-Risk expressed as a positive LOSS in percent."""
    return float(-(mu + quantile_z(1 - conf, nu, dist) * sigma))


def analyse(returns_pct: pd.Series, bundle: dict, v0: float | None = None) -> dict:
    """Everything the Home / Insights pages show, from a window of daily returns."""
    g, m = bundle["garch"], bundle["markov"]
    r = returns_pct.dropna().astype(float)
    v0 = bundle["sample_var"] if v0 is None else v0
    var, nxt = garch_filter(r.values, g["mu"], g["omega"], g["alpha"], g["beta"], v0)
    probs = markov_filter(r.values, m["mu"], m["sigma2"], m["P"])
    hi = int(m["turbulent"])
    p_turb = pd.Series(probs[:, hi], index=r.index, name="p_turbulent")
    sigma_next = float(np.sqrt(nxt))
    p_now = float(p_turb.iloc[-1])
    hist = np.asarray(bundle["train_cond_vol"], float)
    return dict(
        returns=r, cond_vol=pd.Series(np.sqrt(var), index=r.index, name="cond_vol"),
        p_turb=p_turb, score=100 * p_now, confidence=100 * max(p_now, 1 - p_now),
        sigma_next=sigma_next, vol_pctile=float((hist < sigma_next).mean() * 100),
        realised_30d=float(r.tail(30).std()),
        var95=var_pct(sigma_next, g["mu"], g["nu"], 0.95), var99=var_pct(sigma_next, g["mu"], g["nu"], 0.99),
        var95_normal=var_pct(sigma_next, g["mu"], None, 0.95, "normal"),
        var99_normal=var_pct(sigma_next, g["mu"], None, 0.99, "normal"),
    )
