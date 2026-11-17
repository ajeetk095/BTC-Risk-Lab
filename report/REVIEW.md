# REVIEW — defect register for MScFE 610 GWP 2 (Group 8665)

Scope: the original submission (report PDF + code PDF), the first corrected rebuild (v2.0) and this release (v2.1).
Evidence: every item below was reproduced in isolation (`scripts/run_all.py`, the executed notebook and the Bug-lab tab).
Educational project — nothing here is investment advice.

## A. Defects in the original submission (fixed in v2.0)

| # | Area | Defect | Evidence / consequence | Fix |
|---|------|--------|------------------------|-----|
| 1 | Data | `yf.download` returns MultiIndex columns | regressor labelled `BTC-USD`; ECM coefficients printed as tuples | flatten with `get_level_values(0)` |
| 2 | Data | `auto_adjust` default flipped to `True` | every price silently changes; not reproducible | pin `auto_adjust=False`, cache CSVs |
| 3 | Data | no index frequency | repeated `ValueWarning` | `asfreq("D")` for crypto |
| 4 | Data | raw volume (~1e10) mixed with price changes (~1e2) | condition number ≈ 7.7e10; VAR coefficients print as `0.000000` | volume in billions, returns in percent |
| 5 | Stationarity | KPSS p-values printed as exact; two tests described as three | truncated at 0.01 / 0.10 | report bounds; read ADF + KPSS jointly |
| 6 | Cointegration | integration order of the partner series never checked | Volume is I(0) while Close is I(1) | explicit order table; precondition flag |
| 7 | Cointegration | ordinary ADF p-value used on EG residuals | wrong critical values (`coint` imported, never called) | MacKinnon values via `coint` |
| 8 | Cointegration | unbalanced regression (I(1) on I(0)) | verdict should be "precondition violated" | explicit verdict |
| 9 | ECM | ECM fitted although no cointegration | φ = −0.0013, p = 0.35, meaningless | ECM only when valid; GOOGL/GOOG added as a valid example |
| 10 | VAR | lag cap arbitrary (15) and binding | AIC picked the cap | data-driven cap, BIC default |
| 11 | VAR | no diagnostics | whiteness/stability never tested | Portmanteau + stability reported |
| 12 | Regime | `predict()` plotted as a probability | it is the dollar conditional mean (21–59) | use `smoothed_marginal_probabilities` |
| 13 | Regime | dollar differences used | variance grows with price → 2021 "100% turbulent", Mar-2020 "calm" | percent returns |
| 14 | Regime | single-start optimisation | possible local optimum | multi-start, best log-likelihood |
| 15 | Collinearity | VIF computed on the two dependent variables | VIF ≈ 1.001 guaranteed; test vacuous | real predictor set (VIF up to 27.6) |
| 16 | Collinearity | PCA on unstandardised data | "PC1 = 100%" is a units artefact | standardise first |
| 17 | Collinearity | look-ahead in features | R² ≈ 0.59 (leaky) vs 0.008 (lagged) | lag predictors one day |
| 18 | Collinearity | model comparison ignored time order | leakage risk | expanding-window `TimeSeriesSplit` |
| 19 | Features | GARCH on unscaled returns (`DataScaleWarning`) | α=0.673, β=0.036 → "no clustering" (wrong) | percent returns, `rescale=False` |
| 20 | Features | double mean, mislabelled plot (`ARIMA(1,1,1)` vs fitted (1,0,1)), duplicated `show()` | misleading outputs | corrected |

### Report-level issues (text, not code)
- "We assume realistic values" for the regime parameters although the fitted output was available; right-axis values 20–60 read as probabilities.
- Regime 1 described as "significant negative price changes" although its mean is **positive**; variance differences called insignificant although the output is p = 0.000.
- Log-likelihood/HQIC digits mismatch the pasted output; end date quoted as 2025-04-27 in the text but 2025-02-28 in the code; two different GARCH fits quoted in report vs code PDF.
- The conditional-variance plot is flat at ≈0.0099 (the ω floor of a failed fit) and was read as "baseline volatility"; the ACF/PACF show white-noise returns, so the near-cancelling ARMA(1,0,1) terms are not justified.
- 30 pages against a 5–6 page limit; all four challenges covered where a 3-member team selects three; attached "HTML" file is a 553-byte macOS metadata stub, not a notebook export.

## B. Defects found in the corrected version (fixed in v2.1)

| # | Area | Defect | Effect | Fix |
|---|------|--------|--------|-----|
| 21 | Risk | VaR quantile **divided** by the Student-t scale factor | 1-day 95/99% VaR shown as 10.8% / 20.7% (≈2.9× too large); correct ≈ 3.8% / 7.3%. Visible in notebook, app, figure 09, slide 14 | multiply by `sqrt((ν−2)/ν)` |
| 22 | Data | one BTC volume print (2021-02-26: 3.5e11, 5.8× local median, back to normal next day) | −110k residual in the EG regression; ±3e11 spike in ΔVolume | detected (>5× centred 61-day median) and interpolated; raw kept in `Volume_raw` |
| 23 | Regime | narrative "matches the crash timeline" over-claimed | expected durations ≈ 4.1 / 2.3 days, 485 switches → fat-tail mixture, not regimes | `quality` flag + caveat in app, notebook, slides |
| 24 | Features | "IGARCH" read from a boundary solution | `arch` constrains α+β ≤ 1; exactly 1.000 is the bound | `at_boundary` flag; EGARCH-t (0.99, best AIC) cited as the unconstrained check |
| 25 | Collinearity | Ridge grid stopped at 1e4 with the optimum on the edge; headline "remedies improve accuracy" | gains are only 0.04–0.66% | grid widened to 1e6 (optimum is still the maximum ⇒ no usable signal); honest titles |
| 26 | App | `st.stop()` inside one tab | every later tab silently blank | defaults instead of stopping |
| 27 | App/docs | hard-coded verdicts and numbers (look-ahead R² "≈0.57", "peaks in 2020", "could not be reproduced") | claims not tied to the data; the core ADF statistic does reproduce exactly | computed live or re-worded |
| 28 | Figures | IRF figure title clipped | cosmetic | `bbox_inches="tight"` |

## C. Open items / not verified
- GOOGL/GOOG equilibrium gap shows an unexplained mid-2021 excursion (≈ −0.035 log gap for ~3 months) — worth a data check.
- Live CoinGecko calls could not be exercised from the build sandbox (network restricted); the offline fallback path is tested, the live path is not.
- Developer LinkedIn/GitHub URLs are taken from the brief and were not verified.
- VaR has not been back-tested (Kupiec / Christoffersen).
