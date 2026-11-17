import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

from utils.layout import W, footer, setup

ROOT = pathlib.Path(__file__).resolve().parents[1]
setup("Journey · BTC Econometrics Lab", "🧭")
st.title("🧭 Journey")
st.caption("From a coursework submission to a reviewed, corrected and deployed app.")

steps = [
    ("Ideation · Mar 2025", "Assignment brief (Set D)", "MScFE 610 GWP 2: pick challenges from non-stationarity, regime change, multicollinearity and feature extraction, using BTC-USD."),
    ("Data collection · Mar 2025", "Daily BTC-USD from Yahoo Finance", "2020-01-01 → 2025-02-27, close and volume; later cached as CSV so results are reproducible."),
    ("Model training · Mar–Apr 2025", "Original submission (v1.0)", "ADF/KPSS, Engle–Granger, ECM, VAR; Markov switching; VIF/PCA; ARMA + GARCH + Kalman features."),
    ("Review & backtesting", "Defect hunt", "Re-ran everything, tested each suspicion in isolation and found 20 defects — e.g. regime `predict()` read as a probability, unscaled GARCH, look-ahead features."),
    ("Rebuild (v2.0)", "Corrected `src/` package", "Percent returns, lagged predictors, proper regime inference, time-series CV, executed notebook and a Streamlit workbench."),
    ("Streamlit UI & live layer (v2.1)", "This app", "Multi-page UI, CoinGecko live data, fitted-parameter engine, simulation playground — plus a second review pass that fixed the VaR scaling bug, a volume artefact and over-claimed conclusions."),
    ("Deployment", "GitHub → Streamlit Community Cloud", "Public repo, `app.py` as entry point, pinned requirements, offline fallbacks so the app never crashes."),
]
html = '<div class="tl">' + "".join(
    f'<div class="tl-item"><div class="tl-date">{d}</div><div class="tl-title">{t}</div><div>{b}</div></div>' for d, t, b in steps) + "</div>"
st.markdown(html, unsafe_allow_html=True)

st.header("Milestones")
st.markdown("""
- ✅ Core ADF statistic on ΔClose reproduced exactly after the rebuild.
- ✅ 20 defects documented and fixed; second pass added VaR, volume-artefact and over-claim fixes.
- ✅ Re-implemented filters validated against `arch`/`statsmodels` (differences ≈ 1e-5 and 1e-16).
- ⏳ Open item: the GOOGL/GOOG equilibrium gap shows an unexplained mid-2021 excursion worth a data check.
""")

st.header("Figures from the project")
fig = ROOT / "figures"
pairs = [("05_regime.png", "Specification matters: dollar-based vs return-based regimes"),
         ("08_features.png", "Extracted features: conditional mean, GARCH variance, squared returns, Kalman level"),
         ("09_risk.png", "Corrected risk output (volatility forecast and VaR)")]
cols = st.columns(len(pairs))
for c, (f, cap) in zip(cols, pairs):
    if (fig / f).exists():
        c.image(str(fig / f), caption=cap)
footer()
