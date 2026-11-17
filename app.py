"""
BTC Econometrics Lab - Home
Live BTC data + fitted GARCH-t / Markov-switching model -> volatility, turbulence score, VaR.
MScFE 610 Financial Econometrics | Group Work Project 2 | Group 8665
Run:  streamlit run app.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from model.preprocess import to_percent_returns
from utils import engine
from utils.config import APP_NAME, DEV_EMAIL, DEV_MAILTO, TAGLINE
from utils.data_fetch import fetch_history, fetch_prices
from utils.disclaimer import show_compact_disclaimer
from utils.insights import generate_advice
from utils.layout import W, autorefresh, footer, setup
from utils.model_store import load_model

setup(APP_NAME, "📈", banner=False)
autorefresh(60, "home")

# ------------------------------------------------------------------ hero
st.markdown(
    f"""<div class="hero"><h1>📈 {APP_NAME}</h1>
<p><i>{TAGLINE}</i></p>
<p>Non-stationarity &amp; equilibrium · Regime switching · Multicollinearity · Feature extraction</p>
<div class="warn">⚠️ Educational demo only — NOT investment or financial advice.</div></div>""",
    unsafe_allow_html=True)
show_compact_disclaimer()

# ------------------------------------------------------------------ data + model
bundle, model_src = load_model()
prices = fetch_prices()
hist, hmeta = fetch_history(120)
live = prices["source"] == "live" and hmeta["source"] == "live"
rets = to_percent_returns(hist)
a = engine.analyse(rets.tail(90), bundle)

top = st.columns([4, 1])
top[0].caption(f"{prices['note']} · {hmeta['note']} · Last updated {prices['fetched_at']} · auto-refresh every 60 s")
if top[1].button("🔄 Refresh", key="refresh"):
    st.cache_data.clear()
    st.rerun()
if not live:
    st.warning("Live APIs are unreachable right now — the page is showing **cached historical data**, "
               "not the current market.")

# ------------------------------------------------------------------ live stats row
st.subheader("Live model stats")
c = st.columns(5)
c[0].metric("₿ BTC-USD", f"${prices['btc_price']:,.0f}", f"{prices['btc_24h']:+.2f}% 24h")
c[1].metric("📉 30-day realised vol", f"{a['realised_30d']:.2f}%/day")
c[2].metric("🔮 Forecast vol (next day)", f"{a['sigma_next']:.2f}%/day",
            f"{a['sigma_next'] - a['realised_30d']:+.2f} vs realised", delta_color="off")
c[3].metric("🎯 Turbulence score (0–100)", f"{a['score']:.0f}")
c[4].metric("✅ State confidence", f"{a['confidence']:.0f}%")
st.caption("Turbulence score = filtered probability (×100) that today is in the high-variance regime. "
           "Confidence = how decisively the model picks a state. Neither predicts price direction.")

# ------------------------------------------------------------------ live graph
st.subheader("Live model graph — price, turbulence probability and fitted volatility")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="BTC close ($)", line=dict(color="#f7931a", width=2)),
              secondary_y=False)
fig.add_trace(go.Scatter(x=a["p_turb"].index, y=a["p_turb"].values, name="P(turbulent)", line_shape="spline",
                         fill="tozeroy", line=dict(color="#8b5cf6", width=1.2), fillcolor="rgba(139,92,246,.28)"),
              secondary_y=True)
fig.add_trace(go.Scatter(x=a["cond_vol"].index, y=a["cond_vol"].values, name="GARCH vol (%/day)",
                         line=dict(color="#06b6d4", width=1.4, dash="dot"), visible="legendonly"), secondary_y=True)
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="probability / vol (%)", secondary_y=True)
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  height=430, margin=dict(t=30, b=30), legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, **W)

# ------------------------------------------------------------------ insights
st.subheader("Real-time insights")
cards = generate_advice(a, prices["btc_24h"], live)
cols = st.columns(2)
for i, ins in enumerate(cards):
    cols[i % 2].markdown(
        f'<div class="glass insight-{ins["type"]}"><div class="insight-title">{ins["title"]}</div>{ins["body"]}</div>',
        unsafe_allow_html=True)

# ------------------------------------------------------------------ market pulse
st.subheader("Live market pulse")
m = st.columns(4)
m[0].metric("₿ Bitcoin", f"${prices['btc_price']:,.0f}", f"{prices['btc_24h']:+.2f}%")
m[1].metric("Ξ Ethereum", f"${prices['eth_price']:,.0f}", f"{prices['eth_24h']:+.2f}%")
m[2].metric("1-day 95% VaR (t)", f"{a['var95']:.2f}%")
m[3].metric("1-day 99% VaR (t)", f"{a['var99']:.2f}%")
st.caption(f"Model: {model_src}. VaR uses the fitted GARCH(1,1)-t (ν≈{bundle['garch']['nu']:.1f}); "
           "it is a statistical threshold, not a worst case.")

# ------------------------------------------------------------------ CTAs
st.divider()
b = st.columns(4)
b[0].page_link("pages/3_🎛_Simulate.py", label="🎛 Try the simulation", icon=None)
b[1].page_link("pages/2_🎓_Learn.py", label="🎓 Learn how it works")
b[2].page_link("pages/5_📈_Econometrics_Lab.py", label="📈 Open the model workbench")
b[3].link_button("📧 Email the Developer", DEV_MAILTO)

footer()
