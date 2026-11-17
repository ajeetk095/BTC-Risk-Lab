import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

from model.preprocess import to_percent_returns
from utils import engine
from utils.data_fetch import fetch_history, fetch_prices
from utils.insights import daily_tip, generate_advice
from utils.layout import autorefresh, footer, setup
from utils.model_store import load_model

setup("Insights · BTC Econometrics Lab", "📊")
autorefresh(60, "insights")
st.title("📊 Insights")

st.info(f"💡 **Tip of the day:** {daily_tip()}")

bundle, src = load_model()
prices, (hist, meta) = fetch_prices(), fetch_history(120)
live = prices["source"] == "live" and meta["source"] == "live"
a = engine.analyse(to_percent_returns(hist).tail(90), bundle)
st.caption(f"{meta['note']} · {prices['note']}")

st.header("Personalised commentary from current data")
for ins in generate_advice(a, prices["btc_24h"], live):
    {"success": st.success, "info": st.info, "warning": st.warning}[ins["type"]](f"**{ins['title']}** — {ins['body']}")

st.header("Live data + model, side by side")
c = st.columns(3)
c[0].metric("BTC-USD", f"${prices['btc_price']:,.0f}", f"{prices['btc_24h']:+.2f}% 24h")
c[1].metric("Forecast σ (next day)", f"{a['sigma_next']:.2f}%", f"{a['vol_pctile']:.0f}th percentile of 2020–25", delta_color="off")
c[2].metric("Turbulence score", f"{a['score']:.0f} / 100", f"confidence {a['confidence']:.0f}%", delta_color="off")
st.table({
    "Measure": ["30-day realised volatility", "Model next-day volatility", "95% VaR (Student-t)", "95% VaR (normal)",
                "99% VaR (Student-t)", "99% VaR (normal)"],
    "Value (% per day)": [f"{a['realised_30d']:.2f}", f"{a['sigma_next']:.2f}", f"{a['var95']:.2f}", f"{a['var95_normal']:.2f}",
                          f"{a['var99']:.2f}", f"{a['var99_normal']:.2f}"]})
st.caption("At 95% the fat-tailed and normal VaR are similar; the Student-t only becomes clearly larger deeper in the tail (99%).")
st.caption(f"Model parameters: {src}.")
footer()
