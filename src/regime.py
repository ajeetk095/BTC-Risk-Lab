"""Challenge 2 - Detecting a regime change (Hamilton Markov switching).

The single most important fix in the whole project lives here:
``result.predict()`` returns the conditional MEAN of the series in dollars,
NOT a regime label and NOT a probability. Regime inference must use
``smoothed_marginal_probabilities``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .data import quiet

quiet()


def fit_markov(series: pd.Series, k_regimes: int = 2, switching_variance: bool = True,
               search_reps: int = 12, seed: int = 42) -> dict:
    """Maximum-likelihood Markov switching regression with a multi-start search.

    A single default start can settle in a local optimum; we perturb the
    starting values and keep the highest log-likelihood.
    """
    s = series.dropna().astype(float)
    base = MarkovRegression(s, k_regimes=k_regimes, trend="c",
                            switching_variance=switching_variance)
    best, best_ll, tried = None, -np.inf, 0
    rng = np.random.default_rng(seed)

    # Build starting values BY NAME: statsmodels orders them
    # [transitions..., const[0..k], sigma2[0..k]] - guessing the order silently
    # makes every fit fail.
    names = list(base.param_names)
    mu0, sd = float(s.mean()), float(s.std())
    candidates = [None]
    for _ in range(search_reps):
        sv = []
        for nm in names:
            if nm.startswith("p["):
                sv.append(float(rng.uniform(0.80, 0.98)))
            elif nm.startswith("const"):
                sv.append(float(mu0 + rng.normal(0, 0.4 * sd)))
            elif nm.startswith("sigma2"):
                sv.append(float(sd ** 2 * rng.uniform(0.15, 3.0)))
            else:
                sv.append(0.0)
        candidates.append(np.array(sv))

    for sv in candidates:
        try:
            r = base.fit(starting_values=sv, maxiter=500) if sv is not None else base.fit(maxiter=500)
            tried += 1
            ll = float(r.llf)                       # NOT .loglikelihood
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = r, ll
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Markov switching estimation failed for every starting value")

    # ---- correct regime inference ---------------------------------------- #
    # statsmodels returns a DataFrame here; passing it back into pd.DataFrame
    # with new `columns=` would REINDEX and silently produce all-NaN.
    sm = np.asarray(best.smoothed_marginal_probabilities, dtype=float)
    fi = np.asarray(best.filtered_marginal_probabilities, dtype=float)
    smoothed = pd.DataFrame(sm, index=s.index,
                            columns=[f"P_regime_{k}" for k in range(k_regimes)])
    filtered = pd.DataFrame(fi, index=s.index,
                            columns=[f"P_regime_{k}" for k in range(k_regimes)])
    label = smoothed.values.argmax(axis=1)

    # order regimes by fitted variance so "0 = calm, 1 = turbulent" always holds
    variances = np.array([best.params[f"sigma2[{k}]"] if f"sigma2[{k}]" in best.params
                          else best.params["sigma2"] for k in range(k_regimes)])
    order = np.argsort(variances)
    if not np.array_equal(order, np.arange(k_regimes)):
        remap = {int(old): int(new) for new, old in enumerate(order)}
        smoothed = smoothed.rename(columns={f"P_regime_{old}": f"P_regime_{new}"
                                            for old, new in remap.items()})
        smoothed = smoothed[[f"P_regime_{k}" for k in range(k_regimes)]]
        label = np.array([remap[int(v)] for v in label])

    trans = np.zeros((k_regimes, k_regimes))
    for i in range(k_regimes):
        for j in range(k_regimes):
            key = f"p[{i}->{j}]"
            trans[i, j] = float(best.params[key]) if key in best.params else np.nan
    # statsmodels parameterises p[i->j] for j<k; fill the last column by complement
    for i in range(k_regimes):
        row = trans[i]
        if np.isnan(row).any():
            known = np.nansum(row)
            row[np.isnan(row)] = max(0.0, 1.0 - known)
    durations = [float(1.0 / (1.0 - trans[k, k])) if trans[k, k] < 1 else np.inf
                 for k in range(k_regimes)]

    regimes = []
    for k in range(k_regimes):
        mu_key = f"const[{k}]"
        var_key = f"sigma2[{k}]"
        mu = float(best.params[mu_key]) if mu_key in best.params else float(best.params["const"])
        var = float(best.params[var_key]) if var_key in best.params else float(best.params["sigma2"])
        pmu = float(best.pvalues[mu_key]) if mu_key in best.pvalues else np.nan
        regimes.append(dict(regime=k, mu=mu, mu_p=pmu, sigma2=var,
                            sigma=np.sqrt(var), expected_duration=durations[k],
                            share=float((label == k).mean())))

    hi = int(np.argmax([r["sigma2"] for r in regimes]))
    prob_hi = smoothed[f"P_regime_{hi}"]
    switches = int(np.sum(np.diff(label) != 0))

    # Honest labelling: a "regime" that lasts a few days is a fat-tail mixture,
    # not a structural state. Flag it instead of calling it a regime change.
    min_dur = float(min(durations))
    quality = "persistent regimes" if min_dur >= 10 else (
        "semi-persistent" if min_dur >= 5 else "mixture-like (short-lived states)")

    return dict(result=best, series=s, smoothed=smoothed, filtered=filtered,
                label=pd.Series(label, index=s.index, name="regime"),
                regimes=regimes, transition=trans, loglik=float(best.llf),
                aic=float(best.aic), bic=float(best.bic), starts_tried=tried,
                high_vol_regime=hi, prob_high=prob_hi, n_switches=switches,
                min_duration=min_dur, quality=quality,
                predict_range=(float(best.predict().min()), float(best.predict().max())),
                rcm=float(smoothed[f"P_regime_{hi}"].mean() * 100))


def regime_episodes(label: pd.Series, min_len: int = 3) -> pd.DataFrame:
    """Collapse the daily classification into dated episodes."""
    lab = label.values
    change = np.flatnonzero(np.diff(lab) != 0) + 1
    bounds = np.concatenate(([0], change, [len(lab)]))
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= min_len:
            rows.append(dict(start=label.index[a].date(), end=label.index[b - 1].date(),
                             regime=int(lab[a]), days=int(b - a)))
    return pd.DataFrame(rows)
