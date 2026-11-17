"""Challenge 3 - Handling multicollinearity.

The original test was vacuous: VIF was computed on the two *dependent*
variables of the VAR (which are not regressors of one another) and PCA was run
on unstandardised data, so 'PC1 explains 100%' was a units artefact
(volume ~ 1e10 vs price change ~ 1e2), not evidence of collinearity.

Here we build a genuine, leakage-free predictor set (overlapping moving-average
gaps, rolling volatilities, momentum, log volume), all lagged one day, then
demonstrate the symptom and three remedies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data import quiet

quiet()

MA_WINDOWS = (5, 10, 20, 50, 100)
VOL_WINDOWS = (5, 10, 20)


def build_features(df: pd.DataFrame, ma_windows=MA_WINDOWS, vol_windows=VOL_WINDOWS):
    """Return (X, y, frame). Every predictor is lagged 1 day: no look-ahead."""
    c = df["Close"]
    ret = 100.0 * c.pct_change()
    F = pd.DataFrame(index=df.index)
    for w in ma_windows:
        F[f"ma{w}_gap"] = c.rolling(w).mean() / c - 1.0
    for w in vol_windows:
        F[f"vol{w}"] = ret.rolling(w).std()
    F["log_volume"] = np.log(df["Volume"].replace(0, np.nan))
    F["mom5"] = c.pct_change(5)
    F["mom10"] = c.pct_change(10)
    F = F.shift(1)                                   # <- information set at t-1
    joined = pd.concat([ret.rename("y"), F], axis=1).dropna()
    return joined.drop(columns="y"), joined["y"], joined


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF on the predictors only. A constant has no VIF - never report one."""
    Xc = sm.add_constant(X)
    vals = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    out = pd.DataFrame({"feature": X.columns, "VIF": vals})
    out["flag"] = np.where(out.VIF > 10, "severe",
                           np.where(out.VIF > 5, "moderate", "ok"))
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def pca_analysis(X: pd.DataFrame, n_keep: int = 3) -> dict:
    """PCA on the CORRELATION matrix (standardised), not the covariance matrix."""
    Z = StandardScaler().fit_transform(X)
    full = PCA().fit(Z)
    evr = full.explained_variance_ratio_
    k = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    k = min(max(k, 1), X.shape[1])
    reduced = PCA(n_components=n_keep).fit(Z)
    return dict(scaler=StandardScaler().fit(X), pca_full=full, pca_reduced=reduced,
                evr=evr, cum=np.cumsum(evr), n_for_90=k,
                loadings=pd.DataFrame(full.components_[:n_keep].T, index=X.columns,
                                      columns=[f"PC{i+1}" for i in range(n_keep)]),
                cond_raw=float(np.linalg.cond(X.values)),
                cond_scaled=float(np.linalg.cond(Z)), n_keep=n_keep)


RIDGE_GRID = np.logspace(-2, 6, 50)     # widened: the first grid stopped at 1e4 and the optimum sat on that edge


def remedy_comparison(X: pd.DataFrame, y: pd.Series, n_keep: int = 3,
                      n_splits: int = 5) -> dict:
    """In-sample OLS vs. out-of-sample OLS / Ridge / PCR.

    Time-series CV (expanding window) - a random K-fold would leak the future.
    """
    Z = StandardScaler().fit_transform(X)
    ols = sm.OLS(y, sm.add_constant(X)).fit()

    # dropping the worst offenders until every VIF <= 10
    keep = list(X.columns)
    dropped = []
    while True:
        v = vif_table(X[keep])
        if v.empty or v.VIF.max() <= 10 or len(keep) <= 2:
            break
        worst = v.iloc[0]["feature"]
        keep.remove(worst)
        dropped.append(worst)
    ols_trim = sm.OLS(y, sm.add_constant(X[keep])).fit()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    err = {"OLS": [], "Ridge": [], f"PCR{n_keep}": [], "Trimmed OLS": []}
    Zt = StandardScaler().fit_transform(X[keep])
    for tr, te in tscv.split(Z):
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        err["OLS"].append(np.mean((yte - LinearRegression().fit(Z[tr], ytr).predict(Z[te])) ** 2))
        ridge = RidgeCV(alphas=RIDGE_GRID).fit(Z[tr], ytr)
        err["Ridge"].append(np.mean((yte - ridge.predict(Z[te])) ** 2))
        p = PCA(n_components=n_keep).fit(Z[tr])
        err[f"PCR{n_keep}"].append(np.mean(
            (yte - LinearRegression().fit(p.transform(Z[tr]), ytr).predict(p.transform(Z[te]))) ** 2))
        err["Trimmed OLS"].append(np.mean(
            (yte - LinearRegression().fit(Zt[tr], ytr).predict(Zt[te])) ** 2))

    ridge_full = RidgeCV(alphas=RIDGE_GRID).fit(Z, y)
    mse = {k: float(np.mean(v)) for k, v in err.items()}
    base = mse["OLS"]
    gain = {k: 100.0 * (base - v) / base for k, v in mse.items()}
    return dict(ols=ols, ols_trim=ols_trim, kept=keep, dropped=dropped,
                ridge=ridge_full, ridge_alpha=float(ridge_full.alpha_),
                mse=mse, gain_pct=gain, cv_splits=n_splits,
                fold_mse={k: [float(x) for x in v] for k, v in err.items()},
                ridge_at_edge=bool(ridge_full.alpha_ >= RIDGE_GRID[-1] * 0.999),
                folds_better={k: int(sum(a < b for a, b in zip(v, err["OLS"])))
                              for k, v in err.items()},
                n_sig=int((ols.pvalues.drop("const") < 0.05).sum()),
                n_pred=int(X.shape[1]), ols_r2=float(ols.rsquared),
                ols_r2_trim=float(ols_trim.rsquared))
