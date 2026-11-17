"""Shared page chrome: config, CSS, sidebar contact + disclaimer, footer."""
from __future__ import annotations

import inspect

import streamlit as st

from .config import (APP_NAME, COURSE, DEV_EMAIL, DEV_GITHUB, DEV_LINKEDIN, DEV_MAILTO, DEV_NAME)
from .disclaimer import FULL_DISCLAIMER, show_compact_disclaimer, show_footer_disclaimer
from .styles import inject_css

ACCENT = "#f7931a"
GREEN, RED, BLUE = "#2e9e5b", "#d1344b", "#2f6fb5"


def _supports_width() -> bool:
    try:
        return "width" in inspect.signature(st.plotly_chart).parameters
    except Exception:
        return False


# `use_container_width` is deprecated on new Streamlit builds, `width=` is missing on old ones.
W = {"width": "stretch"} if _supports_width() else {"use_container_width": True}


def verdict(text: str, kind: str = "ok") -> None:
    st.markdown(f'<div class="verdict-{kind}">{text}</div>', unsafe_allow_html=True)


def sidebar_common() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="side-disclaimer"><b>⚠️ Educational use only.</b><br>Not financial, investment or '
            'professional advice. Data and outputs may be wrong.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
### 👨‍💻 Connect with the Developer
- 📧 [Email Me]({DEV_MAILTO})
- 💼 [LinkedIn]({DEV_LINKEDIN})
- 💻 [GitHub]({DEV_GITHUB})
""")
        st.caption(f"{APP_NAME} · {COURSE}")
        st.divider()


def setup(page_title: str = APP_NAME, icon: str = "📈", banner: bool = True) -> None:
    """Call first on every page (it must precede any other st.* call)."""
    st.set_page_config(page_title=page_title, page_icon=icon, layout="wide")
    inject_css()
    sidebar_common()
    if banner:
        show_compact_disclaimer()


def autorefresh(seconds: int = 60, key: str = "auto") -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=seconds * 1000, key=key)
    except Exception:       # optional dependency - the app works without it
        pass


def footer() -> None:
    st.divider()
    with st.expander("⚠️ Full disclaimer"):
        st.markdown(FULL_DISCLAIMER)
    st.markdown(
        f'<div class="footer-disclaimer">Built by <b>{DEV_NAME}</b> · <a href="{DEV_MAILTO}">{DEV_EMAIL}</a> · '
        f'<a href="{DEV_LINKEDIN}">LinkedIn</a> · <a href="{DEV_GITHUB}">GitHub</a></div>',
        unsafe_allow_html=True)
    show_footer_disclaimer()
