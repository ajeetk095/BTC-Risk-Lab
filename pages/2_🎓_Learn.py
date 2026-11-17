import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

from utils.layout import footer, setup

setup("Learn · BTC Econometrics Lab", "🎓")
st.title("🎓 Learn")
st.header("How to use the simulation")
st.markdown("""
1. Open **🎛 Simulate**. The sliders start at the model's latest values.
2. Set a **shock** — the return you imagine for today (e.g. −8%) — and see how it feeds tomorrow's volatility.
3. Move **α** (reaction to shocks) and **β** (memory of past volatility) to see why clustering makes risk persistent.
4. Pick a **horizon** and a **VaR confidence level** and watch the volatility path and loss threshold change.
5. Optionally enter the **last few daily returns** to see the regime probability update.
6. Download your scenario as CSV, or read the matching commentary on **📊 Insights**.
""")

st.header("What the numbers tell you")
st.table({
    "Turbulence score": ["0 – 35", "35 – 65", "65 – 100"],
    "Meaning": ["Calm state likely (ordinary daily moves)", "Mixed — states hard to tell apart",
                "High-variance state likely (fat-tail days)"],
    "Sensible reading (educational)": ["Risk indicators are near their usual level",
                                       "Ignore the label; use the volatility forecast",
                                       "Risk metrics are elevated — expect larger swings either way"]})
st.caption("The score says nothing about whether prices go up or down.")

st.header("FAQ")
for q, a in [
    ("Does the app predict Bitcoin's price?", "No. It models how *large* moves are likely to be (volatility, tail risk), not their direction."),
    ("Why percent returns instead of prices?", "Prices have a unit root (they wander); returns are stationary. Using decimals instead of percent also breaks GARCH estimation."),
    ("What is the confidence figure?", "How decisively the regime filter picks a state: max(P, 1−P). It is not a probability that a forecast is right."),
    ("Why does VaR use a Student-t?", "Daily BTC returns have very fat tails (ν≈3). A normal distribution understates the 99% loss."),
    ("What does persistence 1.000 mean?", "α+β sits on the optimiser's upper bound, so volatility shocks decay very slowly. EGARCH, which is unconstrained, gives about 0.99."),
    ("Is the live data reliable?", "It comes from CoinGecko's free API and may be delayed or rate-limited. When it fails the app says so and uses cached data."),
    ("Can I trade on this?", "No. It is an educational project; see the disclaimer."),
]:
    with st.expander(q):
        st.markdown(a)

st.header("Glossary")
st.markdown("""
- **Stationary / I(0)**: mean and variance do not drift. **I(1)**: needs differencing once (a random walk).
- **GARCH(1,1)**: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ — today's variance reacts to yesterday's shock and remembers yesterday's variance.
- **Markov switching**: a hidden state flips between regimes with fixed transition probabilities.
- **VaR**: the loss that is exceeded only (1 − confidence) of the time.
- **VIF**: how much a predictor's variance is inflated by overlap with other predictors (>10 = severe).
""")
footer()
