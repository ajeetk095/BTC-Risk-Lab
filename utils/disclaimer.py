"""Disclaimer components - one wording, reused on every page."""
import streamlit as st

from .config import APP_NAME, DEV_NAME

COMPACT_DISCLAIMER = (
    "⚠️ **Educational demo only** — not financial, investment, trading, legal, tax or medical advice. "
    "Live data may be delayed or wrong; model outputs can be wrong."
)

FULL_DISCLAIMER = f"""
**⚠️ Important Disclaimer**

**{APP_NAME} is an educational and research project only.**

- This application, its econometric models, live data feeds, scores, simulations, charts and any
  "insights" or "advice" it displays are provided **strictly for educational, demonstration and research purposes.**
- **Nothing here constitutes financial, investment, trading, legal, tax, medical or any other form of professional advice.**
- Outputs are **not recommendations to buy, sell, hold or trade** any asset, security, cryptocurrency or financial instrument.
- Live data comes from third-party public APIs and may be delayed, inaccurate, incomplete or unavailable.
  **Do not rely on it for real-world decisions.**
- Model outputs rest on simplified assumptions and a limited sample (daily BTC-USD, 2020-01 to 2025-02). They can be
  **wrong, biased or misleading** — in particular, they do not predict the direction of prices.
- Always consult a **qualified, licensed professional** before making any financial, investment or other important decision.
- The developer (**{DEV_NAME}**) and the project's co-authors accept **no liability** for any loss, damage or decision
  made on the basis of this app's content.

**By using this app, you acknowledge and agree to these terms.**
"""

FOOTER_TEXT = ("Educational use only · Not financial, investment or professional advice · "
               "Data and model outputs may be wrong or delayed.")


def show_compact_disclaimer() -> None:
    st.warning(COMPACT_DISCLAIMER)


def show_full_disclaimer() -> None:
    st.error(FULL_DISCLAIMER)


def show_footer_disclaimer() -> None:
    st.markdown(f'<div class="footer-disclaimer">{FOOTER_TEXT}</div>', unsafe_allow_html=True)
