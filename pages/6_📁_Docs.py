import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import streamlit as st

from utils.config import APP_NAME, VERSION
from utils.layout import footer, setup
from utils.model_store import load_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
setup("Docs · BTC Econometrics Lab", "📁")
st.title("📁 Docs")
st.warning("These documents are educational artefacts from a coursework project. Not investment advice.")

st.header("Downloads")
files = [
    ("📄 Download Report.pdf", ROOT / "assets/docs/Report.pdf", "application/pdf"),
    ("📊 Download Presentation.pptx", ROOT / "assets/docs/Presentation.pptx",
     "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("🧾 Download REVIEW.md (defect register)", ROOT / "report/REVIEW.md", "text/markdown"),
    ("🔢 Download results.json", ROOT / "report/results.json", "application/json"),
    ("📓 Download notebook (HTML)", ROOT / "notebooks/MScFE_610_GWP2_corrected.html", "text/html"),
]
cols = st.columns(3)
for i, (label, path, mime) in enumerate(files):
    if path.exists():
        cols[i % 3].download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=f"dl{i}")
    else:
        cols[i % 3].caption(f"{path.name} not found in this deployment")

bundle, src = load_model()
g, mk = bundle["garch"], bundle["markov"]
st.header("Model card")
st.markdown(f"""
| | |
|---|---|
| **App / version** | {APP_NAME} {VERSION} |
| **Task** | Volatility forecast, 2-state turbulence probability, one-day VaR for BTC-USD (not a price forecast) |
| **Training data** | Daily BTC-USD, {bundle['window'][0]} → {bundle['window'][1]} ({bundle['n_obs']:,} obs), percent returns; one volume artefact interpolated |
| **Volatility model** | {g['spec']}: μ={g['mu']:.3f}, ω={g['omega']:.3f}, α={g['alpha']:.3f}, β={g['beta']:.3f}, ν={g['nu']:.2f} |
| **Regime model** | Hamilton 2-state Gaussian: μ=({mk['mu'][0]:.2f}, {mk['mu'][1]:.2f}), σ²=({mk['sigma2'][0]:.1f}, {mk['sigma2'][1]:.1f}), p₀₀={mk['P'][0][0]:.3f}, p₁₁={mk['P'][1][1]:.3f} |
| **Validation** | Re-implemented filters match `arch` / `statsmodels` to {bundle['validation']['max_abs_gap_cond_vol_pct']:.1e} (vol) and {bundle['validation']['max_abs_gap_filtered_prob']:.1e} (probabilities) |
| **Known limits** | α+β on optimiser bound; regimes are short-lived; no directional skill; crypto-specific sample; third-party live data |
| **Parameter source now** | `{src}` |
""")

st.header("API reference (live data)")
st.table({
    "Provider": ["CoinGecko", "CoinGecko", "Yahoo Finance (offline cache)"],
    "Endpoint": ["/api/v3/simple/price", "/api/v3/coins/bitcoin/market_chart", "data/*.csv (committed)"],
    "Used for": ["Spot BTC & ETH + 24h change (cache 60 s)", "120-day daily closes → returns → filters (cache 10 min)", "Training sample and offline fallback"],
    "Key needed": ["No", "No", "No"]})
st.caption("Open-Meteo / NewsAPI from the generic template are intentionally not used: weather and headlines are irrelevant to a BTC econometrics model.")

st.header("Version history")
st.markdown("""
- **2.1.0** — Multi-page app with live CoinGecko data, simulation playground, insights, docs; fixed the VaR quantile bug (was ≈2.9× too large),
  added a volume-artefact guard, widened the ridge grid, honest regime and IGARCH-boundary caveats, removed a `st.stop()` that blanked later tabs.
- **2.0.0** — Corrected rebuild: percent returns, lagged features, correct regime inference, `src/` package, notebook, Streamlit workbench.
- **1.0.0** — Original MScFE 610 GWP 2 submission (Group 8665).
""")
footer()
