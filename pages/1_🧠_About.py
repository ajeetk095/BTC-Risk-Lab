import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import streamlit as st

from utils.config import (APP_NAME, COURSE, DEV_EMAIL, DEV_GITHUB, DEV_LINKEDIN, DEV_MAILTO, DEV_NAME, TEAM, VERSION)
from utils.disclaimer import show_full_disclaimer
from utils.layout import footer, setup
from utils.model_store import load_model
from utils.avatar import avatar_path

setup(f"About · {APP_NAME}", "🧠")
st.title("🧠 About")

bundle, src = load_model()
g, mk = bundle["garch"], bundle["markov"]
R = json.loads((pathlib.Path(__file__).resolve().parents[1] / "report" / "results.json").read_text())

st.header(f"{APP_NAME}")
st.markdown(f"""
**What it is.** A teaching app built from the *{COURSE}* coursework. It re-runs four econometric
challenges on daily BTC-USD data and then applies the fitted models to **live** prices.

**What it predicts — and what it does not.** It does **not** forecast price direction. It estimates
(1) next-day **volatility** with a GARCH(1,1) model using Student-t errors, (2) the probability that today belongs to a
**high-variance state** with a 2-regime Markov-switching model, and (3) a one-day **Value-at-Risk** threshold.

**Data.** Daily BTC-USD and ETH-USD from Yahoo Finance ({bundle['window'][0]} → {bundle['window'][1]},
{bundle['n_obs']:,} observations, committed as CSV), plus live prices from CoinGecko.
""")
c = st.columns(4)
c[0].metric("Volatility model", g["spec"])
c[1].metric("Student-t ν", f"{g['nu']:.2f}")
c[2].metric("Persistence α+β", f"{g['persistence']:.3f}")
c[3].metric("Regime states", "2 (calm / turbulent)")
st.markdown(f"""
**Honest accuracy notes**
- Heavy tails matter: normal-error GARCH has AIC {max(r['AIC'] for r in R['features']['spec_table']):,.0f}, versus
  {min(r['AIC'] for r in R['features']['spec_table']):,.0f} for the best model (EGARCH-t) — the t-error versions are far better.
- Persistence of exactly 1.000 is the optimiser's upper bound (`arch` imposes α+β ≤ 1); EGARCH, which is unconstrained, gives
  {next(r['persistence'] for r in R['features']['spec_table'] if r['spec'].startswith('EGARCH')):.3f}. Read it as "very persistent".
- Regime states last only ~{mk['expected_duration'][0]:.1f} (calm) and ~{mk['expected_duration'][1]:.1f} (turbulent) days, so they are
  a descriptive flag for fat-tail days rather than structural regimes.
- Predictors of next-day returns carry almost no out-of-sample signal (best remedy beats OLS by {max(v for k, v in R['remedies']['gain_pct'].items() if k != 'OLS'):.1f}%).
- Model parameters source: `{src}`. Version {VERSION}.
""")

st.header("👨‍💻 Developer")
left, right = st.columns([1, 3])
left.image(avatar_path(), caption=DEV_NAME, width=180)
with right:
    st.markdown(f"""
### {DEV_NAME}
*Quantitative finance aspirant · Bangalore, India*

Hi, I'm Ajeet Kumar, a quantitative finance aspiring professional based in Bangalore, India. I hold a **B.Tech in Information Technology**
from Dr. A.P.J. Abdul Kalam Technical University and an **M.Tech in Computer Science and Engineering** from Netaji Subhas University of
Technology, New Delhi (CGPA 7.92). I recently graduated from **WorldQuant University with an M.Sc. in Financial Engineering (88%)**.

My work sits at the intersection of technology and finance. I'm skilled in **Python, C++ and SQL**, and I've spent the last few years applying
quantitative methods — Markowitz MVO, Black-Litterman, Monte Carlo simulation — alongside machine-learning techniques such as Random Forest and deep learning
to real problems.

I've put this into practice through projects in **portfolio optimisation, Fama-French factor analysis and covariance denoising**, and through internships at
**Amul GCMMF** and **Algo8.ai**, where I built AI-driven defect-detection systems using computer vision. Along the way, I won **1st place at the Algo8 ML Hackathon at IIT Kanpur**
and was **felicitated by the President of India for academic excellence**.

I'm now looking to bring this mix of finance and engineering skill to a **full-time role in quantitative research or financial engineering** — I'd love to hear from teams
working on interesting problems in that space.
""")
    b = st.columns(3)
    b[0].link_button("📧 Email", DEV_MAILTO)
    b[1].link_button("💼 LinkedIn", DEV_LINKEDIN)
    b[2].link_button("💻 GitHub", DEV_GITHUB)

st.markdown(f"**Coursework team (Group 8665):** {' · '.join(TEAM)}. The original submission was a group effort; this rebuilt app, its fixes and the live layer are by {DEV_NAME}.")

st.header("Disclaimer")
show_full_disclaimer()
footer()
