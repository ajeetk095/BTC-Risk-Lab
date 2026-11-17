"""Load the fitted model bundle, with a built-in fallback so the app can never crash."""
from __future__ import annotations

import pathlib

import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"
REQUIRED = {"garch", "markov", "sample_var", "train_cond_vol"}

# Copy of the fitted parameters (same numbers as model/model.pkl). Used when the .pkl is
# missing, unreadable, or was written by an incompatible library version.
DEMO_BUNDLE = {'name': 'built-in demo parameters', 'version': '2.1.0', 'trained_on': 'built-in copy of the fitted parameters', 'window': ['2020-01-02', '2025-02-27'], 'n_obs': 1884, 'garch': {'spec': 'GARCH(1,1)-t', 'mu': 0.0962149034810921, 'omega': 0.15072186757157274, 'alpha': 0.06578905441320147, 'beta': 0.9342109455867985, 'nu': 3.1015789426459217, 'persistence': 1.0, 'at_boundary': True}, 'markov': {'mu': [0.07959514644517537, 0.37979883178193047], 'sigma2': [2.7143893723488337, 25.898706752218136], 'P': [[0.755057050584244, 0.244942949415756], [0.4376492036043062, 0.5623507963956937]], 'turbulent': 1, 'quality': 'mixture-like (short-lived states)', 'expected_duration': [4.082583321484553, 2.2849350387579697]}, 'sample_var': 11.059434727698138, 'train_cond_vol': [1.862, 2.27, 2.428, 2.548, 2.66, 2.775, 2.87, 2.945, 3.032, 3.105, 3.274, 3.391, 3.493, 3.618, 3.782, 3.914, 4.114, 4.398, 4.812, 5.506, 9.634], 'validation': {'max_abs_gap_cond_vol_pct': 4.8648279309837505e-05, 'max_abs_gap_filtered_prob': 6.661338147750939e-16}}


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[dict, str]:
    """Return (bundle, source_description)."""
    try:
        import joblib
        b = joblib.load(MODEL_PATH)
        if REQUIRED <= set(b):
            return b, f"model/model.pkl (trained {b.get('trained_at', 'n/a')})"
    except Exception:
        pass
    return DEMO_BUNDLE, "built-in demo parameters (model.pkl not found or unreadable)"
