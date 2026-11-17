"""Build and execute the corrected submission notebook."""
from __future__ import annotations
import pathlib
import nbformat as nbf
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
co = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# MScFE 610 Financial Econometrics — Group Work Project 2
## BTC Econometrics Lab · Group 8665

**Team:** Kow Kai Sheng (Malaysia) · Ajeet Kumar (India) · Faith Nyaboe Momanyi (Kenya)

**Challenges addressed:** (1) non-stationarity and finding an equilibrium, (2) detecting a regime
change, (3) handling multicollinearity, (4) feature extraction.

**Asset:** BTC-USD daily, 2020-01-01 → 2025-02-27 (Yahoo Finance). A genuinely cointegrated
second pair (Alphabet's GOOGL / GOOG dual-class shares) is added so the equilibrium/ECM section
has a valid positive example, not only a rejection.

> This is the **corrected** notebook. Every defect found in the original submission is fixed and
> flagged inline with `FIX:` comments. The companion file `report/REVIEW.md` lists all of them
> with before/after evidence.

---
""")

md("## 0. Setup and reproducible data access")
co("""import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src import data, stationarity as stn, regime as rg, collinearity as mc, features as ft

data.quiet(); warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3,
                     'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.width', 140)
print('numpy', np.__version__, '| pandas', pd.__version__, '| statsmodels', sm.__version__)""")

md("""### FIX 1–4: the data layer

The original `fetch_data()` broke silently in four ways:

| # | defect | consequence | fix |
|---|--------|-------------|-----|
| 1 | `yf.download` now returns **MultiIndex** `('Price','Ticker')` columns | regressor labelled `BTC-USD` instead of `Volume`; ECM coefficients printed as tuples `('lag_eg_resid','')` | flatten with `get_level_values(0)` |
| 2 | `auto_adjust` default flipped to `True` | every price silently changed → results not reproducible | pin `auto_adjust=False` |
| 3 | no index frequency | repeated `ValueWarning: No frequency information` | `asfreq('D')` for crypto |
| 4 | raw `Volume` (~1e10) mixed with price changes (~1e2) | condition number 7.7e10; VAR coefficients print as `0.000000` | volume in **billions**, returns in **percent** |

A committed CSV cache also makes the notebook run with no network at all.
""")
co("""raw, provenance = data.get_data('BTC-USD', 'cache')
d = data.prepare(raw, 'BTC-USD')
print(provenance)
print('observations:', len(d), '| window:', d.index[0].date(), '->', d.index[-1].date())
print('flat columns:', list(d.columns))
d.head(3)""")

md("""---
# 1. Non-stationarity and finding an equilibrium

## Definition

**Augmented Dickey–Fuller** (H₀: a unit root is present)

$$\\Delta y_t=\\alpha+\\beta t+\\phi y_{t-1}+\\sum_{i=1}^{p}\\gamma_i\\Delta y_{t-i}+\\varepsilon_t$$

**KPSS** (H₀: the series is stationary), with $y_t=\\mu+\\delta t+R_t+\\varepsilon_t$ and $R_t$ a random walk

$$\\eta=\\frac{1}{T^{2}}\\cdot\\frac{\\sum_{t=1}^{T}S_t^{2}}{\\hat\\sigma^{2}},\\qquad S_t=\\sum_{i=1}^{t}\\hat\\varepsilon_i$$

**Engle–Granger cointegration:** estimate $y_t=\\alpha+\\beta x_t+\\epsilon_t$ by OLS, then test
$\\hat\\epsilon_t$ for a unit root using **MacKinnon cointegration critical values**.

**Error-correction model**

$$\\Delta y_t=\\alpha+\\phi\\,(y_{t-1}-\\hat\\beta x_{t-1})+\\gamma\\Delta x_t+\\varepsilon_t,\\qquad \\phi<0$$

**VAR(p)**

$$\\mathbf{z}_t=\\mathbf{c}+\\sum_{i=1}^{p}\\mathbf{A}_i\\mathbf{z}_{t-i}+\\mathbf{u}_t$$

## Description
A price level is typically I(1); its first difference or return is I(0). Two I(1) series are
cointegrated when some linear combination is I(0), which defines a long-run equilibrium toward
which the system is pulled. The ECM's $\\phi$ measures the daily speed of that pull.
""")

md("## Demonstration — integration order")
co("""ur = stn.unit_root_table(d, ['Close','lnClose','Volume','lnVolume','dClose','ret_pct','dVol_bn'])
ur""")

md("""### FIX 5: KPSS p-values are bounded, and the two tests must be read jointly
`statsmodels` truncates KPSS p-values to $[0.01, 0.10]$. The original printed `p: 0.1` as if it were
exact; we report the bound. We also classify each series by the **joint** ADF/KPSS outcome.

### FIX 6: the integration order of the *partner* series was never checked
This is decisive. Cointegration is defined between series of the **same** order, normally I(1).
""")
co("""orders = [stn.integration_order(d, 'Close'), stn.integration_order(d, 'Volume')]
pd.DataFrame(orders)""")

md("""## Demonstration — Engle–Granger on BTC close vs volume

### FIX 7: wrong critical values
The original ran a plain `adfuller` on the OLS residuals and read off the ordinary Dickey–Fuller
p-value. Residual-based cointegration tests need MacKinnon's values, which
`statsmodels.tsa.stattools.coint` supplies — a function the original notebook **imported but never
called**.

### FIX 8: unbalanced regression
`Close` is I(1) but `Volume` is **already stationary in levels**, so this is not a cointegration
problem at all. The correct verdict is "precondition violated", not merely "no cointegration".
""")
co("""eg = stn.engle_granger(d, 'Close', 'Volume')
print(eg['ols'].summary())
print(f"\\nPrecondition (both I(1)) satisfied: {eg['precondition_ok']}")
print(f"naive ADF p-value on residuals : {eg['naive_adf_p']:.5f}   <- what the original reported")
print(f"correct Engle-Granger p-value  : {eg['eg_p']:.5f}   <- MacKinnon critical values")
print('EG critical values:', {k: round(v,3) for k,v in eg['eg_crit'].items()})
print(f"R2 = {eg['r2']:.4f} | Durbin-Watson = {eg['dw']:.4f} | cointegrated = {eg['cointegrated']}")""")

md("""## Diagram""")
co("""fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0,0].plot(d.index, d['Close'], lw=.8, color='darkorange')
ax[0,0].set_title('BTC close: I(1), non-stationary'); ax[0,0].set_ylabel('USD')
ax[0,1].plot(d.index, d['ret_pct'], lw=.6, color='steelblue')
ax[0,1].set_title('Daily percent return: I(0), stationary'); ax[0,1].set_ylabel('%')
ax[1,0].scatter(d['Volume']/1e9, d['Close'], s=5, alpha=.4)
ax[1,0].set_xlabel('Volume (billions)'); ax[1,0].set_ylabel('Close (USD)')
ax[1,0].set_title(f"Spurious regression: R2={eg['r2']:.3f}, DW={eg['dw']:.3f}")
ax[1,1].plot(eg['resid'].index, eg['resid'].values, lw=.7, color='crimson')
ax[1,1].axhline(0, color='k', lw=.8)
ax[1,1].set_title('Equilibrium gap: has a unit root'); ax[1,1].set_xlabel('Date')
fig.tight_layout(); plt.show()""")

md("""## Diagnosis
The levels regression is **spurious** in the Granger–Newbold sense: $R^2=0.101$ looks like a
finding, but the Durbin–Watson statistic of $0.074$ is almost zero and the residual carries a unit
root (EG $p=0.957$). The condition number of $7.7\\times10^{10}$ is a units problem, not evidence of
multicollinearity — a single regressor cannot be collinear with itself.

## Demonstration — ECM (and why it must be gated)

### FIX 9: an ECM was fitted without cointegration
By the Granger representation theorem an ECM is only valid if the pair is cointegrated. The
original fitted one anyway and then explained the insignificant $\\phi$ as a modelling failure.
""")
co("""ecm_bad = stn.fit_ecm(eg, d, 'Close', 'Volume')
print(ecm_bad['model'].summary())
print(f"\\nphi = {ecm_bad['phi']:.6f} (p = {ecm_bad['phi_p']:.4f}) -> valid ECM: {ecm_bad['valid']}")""")

md("""## Directions — a pair that genuinely is cointegrated

Alphabet's **GOOGL** (voting) and **GOOG** (non-voting) shares are claims on the *same* cash flows,
so no-arbitrage predicts a log-price hedge ratio near 1 and a stationary spread. This gives the
assignment a valid equilibrium and a working ECM.
""")
co("""g1, _ = data.get_data('GOOGL'); g2, _ = data.get_data('GOOG')
p1, p2 = data.prepare(g1, 'GOOGL'), data.prepare(g2, 'GOOG')
pair = pd.concat([p1['Close'].rename('GOOGL'), p2['Close'].rename('GOOG')], axis=1).dropna()

eg2 = stn.engle_granger(pair, 'GOOGL', 'GOOG', log=True)
ecm2 = stn.fit_ecm(eg2, pair, 'GOOGL', 'GOOG', log=True)
print(f"n = {eg2['n']} | both I(1): {eg2['precondition_ok']}")
print(f"hedge ratio beta = {eg2['beta']:.4f}  (theory predicts ~1)")
print(f"Engle-Granger p  = {eg2['eg_p']:.5f} -> cointegrated: {eg2['cointegrated']}")
print(ecm2['model'].summary())
print(f"\\nphi = {ecm2['phi']:.5f} (p = {ecm2['phi_p']:.2e})")
print(f"{abs(ecm2['phi'])*100:.2f}% of the gap closes per day; half-life = {ecm2['half_life_days']:.1f} days")""")

co("""fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(eg2['resid'].index, eg2['resid'].values, lw=.8, color='seagreen')
ax[0].axhline(0, color='k', lw=.8)
ax[0].set_title(f"ln(GOOGL) - {eg2['beta']:.4f} ln(GOOG): stationary equilibrium gap")
ax[0].set_ylabel('log spread')
ax[1].plot(ecm2['data'].index, ecm2['data']['ec_lag'], lw=.7, label='lagged gap')
ax[1].plot(ecm2['data'].index, ecm2['data']['dY'], lw=.7, alpha=.7, label='change in ln GOOGL')
ax[1].set_title(f"ECM: phi={ecm2['phi']:.4f}, half-life={ecm2['half_life_days']:.1f} days")
ax[1].legend(fontsize=8); ax[1].set_xlabel('Date')
fig.tight_layout(); plt.show()""")

md("""## Demonstration — VAR on stationary, scale-matched differences

### FIX 10: the lag cap was arbitrary and binding
`max_lags = min(15, n-1)` made AIC/HQIC/FPE all select **exactly 15** — the cap, not an interior
optimum. We use the Schwert rule $\\lfloor 12(n/100)^{1/4}\\rfloor$ and prefer the parsimonious BIC.

### FIX 11: no diagnostics were run
We add the stability check, a Portmanteau whiteness test and Granger causality.
""")
co("""var = stn.fit_var(d, ['dClose','dVol_bn'], 'bic')
print('data-driven lag cap :', var['maxlags'])
print('orders by criterion :', var['orders'])
print(f"chosen by BIC       : {var['p']}   (original AIC choice was pinned at the cap)")
print('stable              :', var['stable'])
print(f"whiteness p-value   : {var['whiteness_p']:.3g}")
print('Granger causality   :', var['granger'])
print(var['result'].summary())""")

co("""fig = var['result'].irf(12).plot(orth=True, figsize=(10, 5))
fig.suptitle(f"VAR({var['p']}) orthogonalised impulse responses", y=1.02)
fig.tight_layout(); plt.show()""")

md("""## Diagnosis / Damage / Directions / Deployment (Challenge 1)
* **Diagnosis.** Price levels are I(1); returns are I(0). BTC close and volume are **not**
  cointegrated, and the regression is unbalanced because volume is I(0). The VAR is stable but its
  residuals are **not white** (Portmanteau $p\\approx10^{-9}$), so linear dynamics remain unmodelled.
* **Damage.** Using the levels regression for hedging or forecasting would be a classic spurious-
  regression error. An ECM built on it has no error-correction mechanism.
* **Directions.** Model returns, not dollar levels; use a genuinely cointegrated pair (GOOGL/GOOG)
  for equilibrium work; extend the VAR (or move to a GARCH-in-mean / VARMA specification) until
  residuals are white.
* **Deployment.** The GOOGL/GOOG spread is a live statistical-arbitrage signal: enter when the gap
  exceeds a threshold, expect reversion with a ~12-day half-life, and re-test cointegration monthly
  because the relationship can break.

---
# 2. Detecting a regime change

## Definition
$$y_t=\\mu_{S_t}+\\epsilon_t,\\qquad \\epsilon_t\\sim N(0,\\sigma^2_{S_t}),\\qquad P(S_t=j\\mid S_{t-1}=i)=p_{ij}$$
$$\\mathbb{E}[\\text{duration of regime } i]=\\frac{1}{1-p_{ii}}$$

## Description
An unobserved Markov chain switches the mean and variance between a calm and a turbulent state.
The Hamilton filter/smoother recovers $P(S_t=j\\mid y_1,\\dots,y_T)$.
""")

md("""### FIX 12 — the most serious defect in the submission
The original code did:
```python
df['Regime'] = regime_result.predict()
```
and the report then wrote *"the regime values (20–60) likely correspond to the probability (%) of
being in Regime 1 … a value of 60 implies a 60% probability"*.

**`predict()` returns the conditional mean of the series in dollars**, a probability-weighted
average of the regime means. It is not a label and not a probability. The observed 20–60 range is
simply the two fitted means (21.06 and 58.79). Correct inference uses
`smoothed_marginal_probabilities`.

### FIX 13 — dollar differences make regimes track the price level
BTC rose from ~\\$7k to ~\\$90k, so a constant *percentage* move becomes a much larger *dollar* move.
A regime model on dollar differences therefore confuses the price level with risk.

### FIX 14 — single-start optimisation
Markov likelihoods are multimodal; we run a multi-start search and keep the best log-likelihood.
""")

co("""mk_dollar = rg.fit_markov(d['dClose'])
print('ORIGINAL SPECIFICATION: dollar differences')
print(f"predict() ranges {mk_dollar['predict_range'][0]:.2f} to {mk_dollar['predict_range'][1]:.2f} DOLLARS")
print('  -> these are the regime means, not probabilities')
pd.DataFrame(mk_dollar['regimes'])""")

co("""mk = rg.fit_markov(d['ret_pct'])
print('CORRECTED SPECIFICATION: percent returns')
print(f"log-likelihood {mk['loglik']:.2f} | AIC {mk['aic']:.2f} | starts evaluated {mk['starts_tried']}")
pd.DataFrame(mk['regimes'])""")

co("""k = 2
T = pd.DataFrame(mk['transition'], index=[f'from {i}' for i in range(k)], columns=[f'to {j}' for j in range(k)])
print('Transition matrix:'); print(T.round(4))
hi, lo = mk['high_vol_regime'], 1 - mk['high_vol_regime']
ratio = mk['regimes'][hi]['sigma'] / mk['regimes'][lo]['sigma']
print(f"\\nTurbulent regime is {ratio:.2f}x more volatile than the calm regime")
print(f"regime switches detected: {mk['n_switches']}")""")

md("## Diagram — correct probability inference, and the specification artefact")
co("""fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for a, (m, ttl, col, yl) in zip(ax, [
        (mk, 'Corrected: percent returns', 'seagreen', 'Return (%)'),
        (mk_dollar, 'Original: dollar differences (price-level artefact)', 'crimson', 'Price change ($)')]):
    a.plot(m['series'].index, m['series'].values, lw=.5, color='0.55')
    a.set_ylabel(yl); a.set_title(ttl, fontsize=10)
    a2 = a.twinx(); a2.fill_between(m['prob_high'].index, 0, m['prob_high'].values, color=col, alpha=.4)
    a2.axhline(.5, color='k', ls='--', lw=.8)
    a2.set_ylim(0, 1); a2.set_ylabel('P(turbulent)'); a2.grid(False)
ax[1].set_xlabel('Date')
fig.suptitle('Smoothed regime probabilities', y=.995)
fig.tight_layout(); plt.show()""")

co("""def share_by_year(m):
    s = pd.Series((m['label'] == m['high_vol_regime']).astype(int).values, index=m['label'].index)
    return s.groupby(s.index.year).mean()

cmp_year = pd.DataFrame({'dollar differences (original)': share_by_year(mk_dollar),
                         'percent returns (corrected)': share_by_year(mk)})
print('Share of days classified as turbulent, by year:')
print(cmp_year.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(cmp_year)); w = .38
ax.bar(x-w/2, cmp_year.iloc[:,0], w, label='Dollar differences (original)', color='crimson')
ax.bar(x+w/2, cmp_year.iloc[:,1], w, label='Percent returns (corrected)', color='seagreen')
ax.set_xticks(x); ax.set_xticklabels(cmp_year.index)
ax.set_ylabel('Share of turbulent days'); ax.legend(fontsize=8)
ax.set_title('Dollar-based regimes track the price level, not risk')
fig.tight_layout(); plt.show()""")

md("""## Diagnosis
The dollar specification calls **2021 turbulent on 100% of days and 2024–25 on ~95%**, while
labelling the March-2020 crash calm on 93% of days. That is economically backwards and is purely a
price-level artefact. On percent returns the turbulent share is 25% in 2020, peaks at **44% in 2021** and is
lowest in **2023 (13%)**.

**Caveat (second review pass).** The fitted chain has expected durations of only ~4.1 (calm) and
~2.3 (turbulent) days and switches 485 times in 1,884 days. At daily frequency a two-state model is
mostly separating ordinary days from fat-tail days — a mixture of normals, i.e. volatility clustering —
not a set of persistent structural regimes. The yearly shares are a descriptive flag; GARCH is the
better tool for forecasting risk.

The corrected model gives $\\sigma_{\\text{calm}}=1.65\\%$/day and $\\sigma_{\\text{turb}}=5.09\\%$/day,
a 3.1× ratio, with expected durations of about 4.1 and 2.3 days.

Two further report contradictions are resolved: the turbulent regime has a **higher** fitted mean
(+0.38% vs +0.08%), not "significant negative price changes"; and both variance parameters are
significant at $p<0.001$, so the claim that "variance differences do not show statistical
significance" is incorrect.

## Damage
Sizing positions from dollar-based regimes would have cut risk in the calm 2020 window and kept
full risk through 2024–25 — the opposite of what the volatility actually warranted.

## Directions
Model returns; allow 3 regimes or a switching mean-and-variance specification; add macro covariates;
validate against realised volatility rather than the model's own labels.

## Deployment
Publish the daily smoothed $P(\\text{turbulent})$; de-risk when it crosses 0.5 and hold the reduced
exposure for the expected regime duration (~2–4 days), re-estimating monthly.
""")

co("""episodes = rg.regime_episodes(mk['label'], min_len=3)
print(f'{len(episodes)} turbulent/calm episodes lasting >= 3 days; first 12:')
episodes.head(12)""")

md("""---
# 3. Handling multicollinearity

## Definition
$$\\mathrm{VIF}_j=\\frac{1}{1-R_j^{2}}$$
where $R_j^2$ comes from regressing predictor $j$ on all the other predictors. VIF > 10 is severe.

## Description
Near-duplicate predictors inflate coefficient standard errors, so individual effects become
insignificant and unstable even when the fit looks acceptable.

### FIX 15 — the original test could not detect anything
VIF was computed on `Diff_Close` and `Diff_Volume`: the two **dependent** variables of the VAR,
which are not regressors of one another. VIF ≈ 1.001 was guaranteed by construction. A VIF was also
reported for the intercept, which is meaningless.

### FIX 16 — PCA on unstandardised data
Volume (~1e10) dwarfs price changes (~1e2), so "PC1 explains 100%" was a **units artefact**. The
report then contradicted itself, claiming both "one dominant factor showing alignment" and "no
redundant information". Standardising removes the artefact.

### FIX 17 — look-ahead bias
A moving-average gap built as `rolling(w).mean()/Close - 1` contains today's return, because today's
close is in the denominator. Every predictor here is lagged one day.
""")

co("""Z = d[['dClose','dVol_bn']].dropna()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
ev_raw = PCA().fit(Z.values).explained_variance_ratio_
ev_std = PCA().fit(StandardScaler().fit_transform(Z)).explained_variance_ratio_
print('Original pair, correlation:', round(Z.corr().iloc[0,1], 5))
print('PCA unstandardised (original):', ev_raw)
print('PCA standardised   (fixed)   :', ev_std.round(4))
print('-> no common factor exists; the 100% was a scaling artefact')""")

co("""X, y, frame = mc.build_features(d)
vif = mc.vif_table(X)
print(f'{X.shape[1]} lagged predictors, {len(y)} observations')
vif""")

co("""pca = mc.pca_analysis(X)
print(f"condition number: raw {pca['cond_raw']:.3g} -> standardised {pca['cond_scaled']:.3g}")
print('explained variance ratio:', pca['evr'].round(4))
print('cumulative              :', pca['cum'].round(4))
print(f"{pca['n_for_90']} components reach 90% of the variance")
pca['loadings'].round(3)""")

md("## Diagram")
co("""fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cols = ['crimson' if f=='severe' else ('darkorange' if f=='moderate' else 'seagreen') for f in vif['flag']]
ax[0].barh(vif['feature'], vif['VIF'], color=cols); ax[0].axvline(10, color='k', ls='--', lw=1)
ax[0].invert_yaxis(); ax[0].set_xlabel('VIF'); ax[0].set_title('Severe multicollinearity (VIF > 10)')
ax[1].bar(range(1,len(pca['evr'])+1), pca['evr'], color='steelblue', alpha=.85, label='individual')
ax[1].plot(range(1,len(pca['cum'])+1), pca['cum'], 'o-', color='crimson', label='cumulative')
ax[1].axhline(.9, color='k', ls='--', lw=1)
ax[1].set_xlabel('Principal component'); ax[1].set_ylabel('Explained variance ratio')
ax[1].set_title('Standardised PCA'); ax[1].legend(fontsize=8)
fig.tight_layout(); plt.show()""")

md("""## Demonstration — do the remedies help out of sample?

### FIX 18 — model comparison must respect time ordering
A random K-fold split leaks future information into the training set. We use an expanding-window
`TimeSeriesSplit`.
""")
co("""rem = mc.remedy_comparison(X, y)
ols = rem['ols']
print(f"full OLS: R2 = {ols.rsquared:.5f}, adj R2 = {ols.rsquared_adj:.5f}")
print(f"significant predictors at 5%: {rem['n_sig']} of {rem['n_pred']}")
print(f"trimmed model drops {rem['dropped']}; R2 = {rem['ols_r2_trim']:.5f}")
print(f"Ridge penalty chosen by CV: alpha = {rem['ridge_alpha']:.4g}")
pd.DataFrame({'OOS MSE': rem['mse'], 'improvement vs OLS (%)': rem['gain_pct']}).round(4)""")

co("""print(ols.summary())""")

co("""fig, ax = plt.subplots(figsize=(7.5, 4))
names = list(rem['mse']); vals = [rem['mse'][k] for k in names]
bars = ax.bar(names, vals, color=['crimson','seagreen','steelblue','darkorange'])
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Out-of-sample MSE'); ax.set_title('Remedies vs OLS: out-of-sample MSE (lower is better; gaps are small)')
fig.tight_layout(); plt.show()""")

md("""## Diagnosis / Damage / Directions / Deployment (Challenge 3)
* **Diagnosis.** With a realistic predictor set the pathology is unmistakable: `ma10_gap` VIF ≈ 27.6,
  `ma20_gap` ≈ 18.6, `ma50_gap` ≈ 12.5. Only a minority of coefficients is significant and in-sample
  $R^2$ is below 1%.
* **Damage.** Individually insignificant, sign-unstable coefficients invite false narratives about
  which indicator "drives" returns. OLS is nominally the worst of the four candidates, but the best remedy
  beats it by only 0.04–0.66% out of sample, and Ridge picks the largest penalty on its (widened) grid —
  it shrinks every slope towards zero. Fixing collinearity does not create predictability here.
* **Directions.** Standardise before PCA; trim redundant windows; use Ridge or principal-component
  regression. Three components already carry ~85% of the predictor variance.
* **Deployment.** Ship the regularised model, not the kitchen-sink OLS. Re-check VIF whenever a
  feature is added, and re-select the Ridge penalty on a rolling window.

---
# 4. Feature extraction

## Definition
Conditional mean (ARMA(p,q)):
$$r_t=c+\\sum_{i=1}^{p}\\phi_i r_{t-i}+\\sum_{j=1}^{q}\\theta_j\\varepsilon_{t-j}+\\varepsilon_t$$
Conditional variance (GARCH(1,1)):
$$\\sigma_t^{2}=\\omega+\\alpha\\varepsilon_{t-1}^{2}+\\beta\\sigma_{t-1}^{2},\\qquad \\omega>0,\\ \\alpha,\\beta\\ge0$$
Persistence $\\alpha+\\beta<1$ is required for a finite unconditional variance
$\\bar\\sigma^2=\\omega/(1-\\alpha-\\beta)$; the volatility half-life is $\\ln(0.5)/\\ln(\\alpha+\\beta)$.
Squared return $r_t^2$ is a noisy but unbiased proxy for realised variance.
Kalman local level: $r_t=\\mu_t+\\varepsilon_t$, $\\mu_t=\\mu_{t-1}+\\eta_t$.

## Description
Raw prices are noisy and non-stationary. Returns, an ARMA conditional mean, a GARCH conditional
variance, squared returns and a Kalman-smoothed level compress the series into a few informative,
comparably-scaled features.

### FIX 19 — scaling decided the economic conclusion
`arch` warned `DataScaleWarning: the scale of y is 0.001099`. The original ignored it and obtained
$\\alpha=0.673$, $\\beta=0.036$ (persistence 0.71), then concluded *"baseline level volatility … the
GARCH has not fully captured the volatility spikes"*. On **percent returns** with `rescale=False`,
every t-error specification is extremely persistent (0.99–1.00; normal-error GARCH 0.97) — shocks to
volatility decay very slowly. Note that `arch` constrains α+β ≤ 1, so the exact 1.000 for GARCH-t is the
optimiser sitting on its bound; the unconstrained EGARCH-t (persistence 0.99, best AIC) is the better
evidence. The original conclusion was an optimiser artefact.

### FIX 20 — double mean, mislabelled plot, duplicated `show()`
The original fitted ARIMA, then fitted `arch_model` on its residuals, which adds a *second* constant
mean. The plot title said `ARIMA(1,1,1)` while the model was `(1,0,1)`, and a duplicated
`plt.tight_layout(); plt.show()` emitted an empty figure.
""")

co("""fp = ft.feature_panel(d['ret_pct'], (1,0,1), 'GARCH(1,1)-t')
print(f"ADF on returns: stat {fp['adf'][0]:.3f}, p {fp['adf'][1]:.3g} -> stationary")
print(fp['arma']['result'].summary())""")

co("""g = fp['garch']
print(g['result'].summary())
print(f"\\npersistence alpha+beta = {g['persistence']:.6f}")
print(f"IGARCH (persistence >= 0.995): {g['integrated']}")
print(f"volatility half-life: {g['half_life_days']}")
print(f"tail index nu = {g['nu']:.3f}  (nu < 5 => very heavy tails)")""")

md("## Demonstration — specification race on identical, correctly scaled data")
co("""ft.compare_specs(d['ret_pct'])""")

md("## Diagram")
co("""P = fp['panel']
fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
axs[0].plot(P.index, P['return_pct'], lw=.5, color='0.65', label='Return (%)')
axs[0].plot(P.index, P['conditional_mean'], lw=1, color='navy', label='Conditional mean ARMA(1,0,1)')
axs[0].set_ylabel('Mean (%)'); axs[0].legend(fontsize=7, loc='upper left')
axs[1].plot(P.index, P['conditional_variance'], lw=.9, color='darkorange', label='Conditional variance GARCH(1,1)-t')
axs[1].set_ylabel('Variance (%²)'); axs[1].legend(fontsize=7, loc='upper left')
axs[2].plot(P.index, P['squared_return'], lw=.5, color='crimson', label='Squared return')
axs[2].set_ylabel('Squared return'); axs[2].legend(fontsize=7, loc='upper left')
axs[3].plot(P.index, P['return_pct'], lw=.5, color='0.75', label='Return (%)')
axs[3].plot(P.index, P['kalman_level'], lw=1.4, color='green', label='Kalman smoothed local level')
axs[3].set_ylabel('Level (%)'); axs[3].set_xlabel('Date'); axs[3].legend(fontsize=7, loc='upper left')
fig.suptitle('Extracted features — BTC daily percent returns', y=.995)
fig.tight_layout(); plt.show()   # single show(): no empty duplicate figure""")

co("""k = fp['kalman']
print(f"sigma2.level     = {k['sigma2_level']:.4g}")
print(f"sigma2.irregular = {k['sigma2_irregular']:.4g}")
print(f"signal-to-noise  = {k['signal_to_noise']:.4g} -> {k['interpretation']}")
print(f"corr(squared return, conditional variance) = {fp['sq_vs_garch_corr']:.4f}")
print('\\nThe GARCH variance is a smoothed risk proxy; the squared return is the noisy raw version.')
P.head()""")

md("## Demonstration — deployable risk output")
co("""fc = ft.forecast_risk(g, 10)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fc['horizon_days'], fc['vol_forecast_pct'], 'o-', label='Volatility forecast (%)')
ax.plot(fc['horizon_days'], -fc['VaR95_pct'], 's--', label='95% VaR (loss, %)')
ax.plot(fc['horizon_days'], -fc['VaR99_pct'], '^--', label='99% VaR (loss, %)')
ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Percent'); ax.legend(fontsize=8)
ax.set_title('Volatility term structure and Value-at-Risk')
fig.tight_layout(); plt.show()
fc""")

md("""## Diagnosis / Damage / Directions / Deployment (Challenge 4)
* **Diagnosis.** Returns are stationary (ADF $p\\approx3\\times10^{-26}$). Volatility is
  **near-integrated**: $\\alpha+\\beta=1.000$ for GARCH-t (the optimiser's bound) and 0.99 for EGARCH-t, so the
  unconditional variance is effectively undefined and shocks fade very slowly. The Student-t tail index $\\nu\\approx3.1$ confirms very heavy tails;
  EGARCH-t wins on AIC. The Kalman signal-to-noise ratio is ~1e-5, i.e. no persistent drift.
* **Damage.** With $\\nu\\approx3$ a normal model understates the 99% tail (≈6.4% vs ≈7.3% one-day VaR); at 95%
  the Student-t VaR (≈3.8%) is actually *smaller* than the normal one (≈4.5%) because the standardised t
  concentrates more mass near the centre. Near-integration implies variance forecasts barely
  mean-revert, so long-horizon risk cannot be anchored to a historical
  average. The earlier "no volatility clustering" reading was simply wrong.
* **Directions.** Prefer EGARCH-t on AIC; consider a component/FIGARCH model to separate transitory
  from permanent volatility; re-estimate on a rolling window; validate VaR with Kupiec and
  Christoffersen backtests.
* **Deployment (illustrative, not advice).** Publish the daily volatility term structure and t-based
  95%/99% VaR; monitor breaches of the 1-day 99% VaR as a model-health check.

## Second review pass (v2.1) — defects found *in the corrected version*
| # | defect | effect | fix |
|---|--------|--------|-----|
| 21 | VaR quantile **divided** by the t scale factor instead of multiplying | 1-day 95%/99% VaR shown as 10.8% / 20.7% — about 2.9× too large (correct ≈ 3.8% / 7.3%) | `features.forecast_risk` now multiplies; filters re-validated |
| 22 | One BTC volume print (2021-02-26, 5.8× its local median, reverts next day) | −110k residual in the Engle–Granger regression and a ±3e11 spike in ΔVolume | detected and interpolated in `data.prepare` (raw kept in `Volume_raw`) |
| 23 | Regime narrative over-claimed ("matches the crash timeline") | 2-day / 4-day states are a fat-tail mixture, not regimes | caveat + `quality` flag in `regime.fit_markov` |
| 24 | "IGARCH" read from a boundary solution | `arch` constrains α+β ≤ 1; exactly 1.000 is the bound | `at_boundary` flag; EGARCH 0.99 cited as the unconstrained check |
| 25 | Ridge grid stopped at 1e4 and the optimum sat on the edge | "remedies improve accuracy" headline over-stated | grid widened to 1e6; honest title; the optimum is still the maximum → predictors carry no signal |
| 26 | `st.stop()` inside one Streamlit tab | blanked every later tab | replaced by defaults |

---
## Reproducibility
""")
co("""import numpy, pandas, statsmodels, arch, sklearn, matplotlib
print('numpy       ', numpy.__version__)
print('pandas      ', pandas.__version__)
print('statsmodels ', statsmodels.__version__)
print('arch        ', arch.__version__)
print('scikit-learn', sklearn.__version__)
print('matplotlib  ', matplotlib.__version__)
print('\\ndata source :', provenance)
print('seed        : 42')
print('All results regenerate with:  python scripts/run_all.py')""")

md("""## References (MLA)
Box, George E. P., et al. *Time Series Analysis: Forecasting and Control*. 5th ed., Wiley, 2015.

Dickey, David A., and Wayne A. Fuller. "Distribution of the Estimators for Autoregressive Time Series
with a Unit Root." *Journal of the American Statistical Association*, vol. 74, no. 366a, 1979, pp. 427–431.

Engle, Robert F., and Clive W. J. Granger. "Co-integration and Error Correction: Representation,
Estimation, and Testing." *Econometrica*, vol. 55, no. 2, 1987, pp. 251–276.

Granger, Clive W. J., and Paul Newbold. "Spurious Regressions in Econometrics." *Journal of
Econometrics*, vol. 2, no. 2, 1974, pp. 111–120.

Hamilton, James D. *Time Series Analysis*. Princeton University Press, 1994.

Hamilton, James D. "A New Approach to the Economic Analysis of Nonstationary Time Series and the
Business Cycle." *Econometrica*, vol. 57, no. 2, 1989, pp. 357–384.

Johansen, Søren. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector
Autoregressive Models." *Econometrica*, vol. 59, no. 6, 1991, pp. 1551–1580.

Jolliffe, Ian T. *Principal Component Analysis*. 2nd ed., Springer, 2002.

Kwiatkowski, Denis, et al. "Testing the Null Hypothesis of Stationarity against the Alternative of a
Unit Root." *Journal of Econometrics*, vol. 54, no. 1–3, 1992, pp. 159–178.

WorldQuant University. "Correlation and Multicollinearity." *Financial Econometrics*, Module 1,
Lesson 2, 2025. Lecture.
""")

nb["cells"] = C
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "title": "MScFE 610 GWP2 - BTC Econometrics Lab (corrected) - Group 8665",
})

path = NB / "MScFE_610_GWP2_corrected.ipynb"
nbf.write(nb, path)
print("built", path, "cells:", len(C))

client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(nb, path)
errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
        if o.get("output_type") == "error"]
print("executed. error outputs:", len(errs))
for e in errs[:5]:
    print("  ", e.get("ename"), e.get("evalue"))
print("saved executed notebook ->", path)
