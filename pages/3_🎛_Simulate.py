import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.preprocess import to_percent_returns
from utils import engine
from utils.data_fetch import fetch_history
from utils.insights import generate_advice
from utils.layout import W, footer, setup
from utils.model_store import load_model

setup("Simulate · BTC Econometrics Lab", "🎛")
st.title("🎛 Simulate — live model playground")

bundle, src = load_model()
g, mk = bundle["garch"], bundle["markov"]
hist, meta = fetch_history(120)
rets = to_percent_returns(hist)
base = engine.analyse(rets.tail(90), bundle)
_, v_now = engine.garch_filter(rets.tail(90).values, g["mu"], g["omega"], g["alpha"], g["beta"], bundle["sample_var"])
st.caption(f"Starting point: {meta['note']} Parameters: {src}.")

left, right = st.columns([1, 2])
with left:
    st.subheader("Inputs")
    sigma0 = st.slider("Current daily volatility σ (%)", 0.5, 10.0, float(round(np.sqrt(v_now), 2)), 0.05,
                       help="Defaults to the model's latest conditional volatility.")
    shock = st.slider("Today's return shock (%)", -30.0, 30.0, 0.0, 0.5)
    horizon = st.slider("Horizon (days)", 1, 30, 10)
    conf = st.slider("VaR confidence (%)", 90.0, 99.9, 99.0, 0.1)
    dist = st.radio("Return distribution", ["Student-t (fitted ν)", "Normal"], horizontal=True)
    with st.expander("Stress the model parameters"):
        alpha = st.slider("α  (reaction to shocks)", 0.00, 0.50, float(round(g["alpha"], 3)), 0.005)
        beta = st.slider("β  (memory of variance)", 0.00, 0.99, float(round(g["beta"], 3)), 0.005)
        nu = st.slider("ν  (tail thickness, lower = fatter)", 2.2, 30.0, float(round(g["nu"], 2)), 0.1)
        if alpha + beta > 1:
            st.warning("α+β > 1 makes the variance explode (non-stationary). Shown for teaching only.")
    st.markdown("**Optional — last daily returns (%)**, oldest → newest")
    default_last = [float(round(x, 2)) for x in rets.tail(5).values]
    last = [st.number_input(f"Day t−{4 - i}" if i < 4 else "Today (t)", value=default_last[i], step=0.5, key=f"r{i}")
            for i in range(5)]

# ---- scenario maths: today's variance = sigma0^2, shock feeds tomorrow
omega = g["omega"]
var_next = omega + alpha * (shock - g["mu"]) ** 2 + beta * sigma0 ** 2
path = engine.forecast_variance(var_next, omega, alpha, beta, horizon)
vol = np.sqrt(np.maximum(path, 1e-12))
cum = np.sqrt(np.cumsum(path))
use_t = dist.startswith("Student")
var1 = engine.var_pct(vol[0], g["mu"], nu, conf / 100, "t" if use_t else "normal")
varh = engine.var_pct(vol[-1], g["mu"], nu, conf / 100, "t" if use_t else "normal")

# regime probability from the 5 returns the visitor sees
probs = engine.markov_filter(np.array(last), mk["mu"], mk["sigma2"], mk["P"])[:, mk["turbulent"]]
score, confd = 100 * probs[-1], 100 * max(probs[-1], 1 - probs[-1])

with right:
    st.subheader("Live prediction")
    m = st.columns(4)
    m[0].metric("Turbulence score", f"{score:.0f} / 100")
    m[1].metric("State confidence", f"{confd:.0f}%")
    m[2].metric("Tomorrow's σ", f"{vol[0]:.2f}%", f"{vol[0] - sigma0:+.2f} vs today", delta_color="off")
    m[3].metric(f"1-day {conf:.1f}% VaR", f"{var1:.2f}%")

    st.markdown("**What drives tomorrow's variance** (feature contributions, %²)")
    parts = {"Baseline ω": omega, "Today's shock  α·ε²": alpha * (shock - g["mu"]) ** 2, "Memory  β·σ²": beta * sigma0 ** 2}
    fc = go.Figure(go.Bar(x=list(parts.values()), y=list(parts.keys()), orientation="h",
                          marker_color=["#06b6d4", "#f59e0b", "#8b5cf6"],
                          text=[f"{v:.2f}" for v in parts.values()], textposition="outside"))
    fc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     height=230, margin=dict(t=10, b=10, l=10, r=40), xaxis_title="contribution to σ²ₜ₊₁")
    st.plotly_chart(fc, **W)

    pf = go.Figure()
    pf.add_trace(go.Scatter(x=list(range(1, horizon + 1)), y=vol, name="daily σ (%)", mode="lines+markers", line=dict(color="#8b5cf6")))
    pf.add_trace(go.Scatter(x=list(range(1, horizon + 1)), y=cum, name="cumulative σ over h days (%)", mode="lines+markers",
                            line=dict(color="#06b6d4", dash="dot")))
    pf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300,
                     margin=dict(t=30, b=30), xaxis_title="days ahead", yaxis_title="percent", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(pf, **W)

# ---- advice
st.subheader("Live insight")
sim = dict(base)
sim.update(score=score, confidence=confd, sigma_next=float(vol[0]),
           vol_pctile=float((np.asarray(bundle["train_cond_vol"]) < vol[0]).mean() * 100),
           var95=engine.var_pct(vol[0], g["mu"], nu, .95), var99=engine.var_pct(vol[0], g["mu"], nu, .99),
           var95_normal=engine.var_pct(vol[0], g["mu"], None, .95, "normal"),
           var99_normal=engine.var_pct(vol[0], g["mu"], None, .99, "normal"))
for ins in generate_advice(sim, None, meta["source"] == "live")[:3]:
    {"success": st.success, "info": st.info, "warning": st.warning}[ins["type"]](f"**{ins['title']}** — {ins['body']}")

# ---- history (session state)
if "history" not in st.session_state:
    st.session_state.history = []
if st.button("➕ Save this scenario to history"):
    st.session_state.history.append(dict(time=dt.datetime.now().strftime("%H:%M:%S"), sigma0=sigma0, shock=shock, alpha=alpha, beta=beta, nu=nu,
                                         horizon=horizon, var_conf=conf, dist=dist, sigma_next=vol[0], var_1d=var1,
                                         var_at_horizon=varh, turbulence_score=score, confidence=confd))
hist_df = pd.DataFrame(st.session_state.history)
st.subheader("Scenario history")
if hist_df.empty:
    st.caption("Nothing saved yet — adjust the sliders and press the button above.")
else:
    hf = go.Figure(go.Scatter(x=hist_df["time"], y=hist_df["var_1d"], mode="lines+markers", line=dict(color="#f59e0b")))
    hf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240,
                     margin=dict(t=20, b=30), yaxis_title="1-day VaR (%)")
    st.plotly_chart(hf, **W)
    st.dataframe(hist_df, hide_index=True, **W)
current = pd.DataFrame([dict(sigma0=sigma0, shock=shock, alpha=alpha, beta=beta, nu=nu, horizon=horizon, var_conf=conf, dist=dist,
                             sigma_next=vol[0], var_1d=var1, var_at_horizon=varh, turbulence_score=score, confidence=confd)])
st.download_button("⬇ Download current scenario (CSV)", current.to_csv(index=False).encode(), "scenario.csv", "text/csv")
with st.expander("Full disclaimer"):
    st.markdown("Educational simulation only. Not financial, investment or professional advice — see the footer for the full text.")
footer()
