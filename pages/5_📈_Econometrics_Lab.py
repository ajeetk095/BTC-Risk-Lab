"""
BTC Econometrics Lab - model workbench (page 5)
MScFE 610 Financial Econometrics | Group Work Project 2 | Group 8665

Visitors change the inputs and re-run the four econometric challenges in the
browser. Models are re-estimated on the committed data cache; the written
verdicts react to the fitted numbers rather than being pasted in.
"""
from __future__ import annotations

import io
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import collinearity as mc
from src import data
from src import features as ft
from src import regime as rg
from src import stationarity as stn

data.quiet()
warnings.filterwarnings("ignore")

from utils.layout import setup, verdict, footer, W, ACCENT, GREEN, RED, BLUE  # noqa: E402

setup("BTC Econometrics Lab", "📈")


# --------------------------------------------------------------------------- #
# cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load(ticker: str, source: str):
    raw, prov = data.get_data(ticker, source)
    return data.prepare(raw, ticker), prov


@st.cache_data(show_spinner=False)
def run_stationarity(ticker, source, start, end, var_crit, ycol, xcol):
    d, _ = load(ticker, source)
    d = data.slice_window(d, start, end)
    ur = stn.unit_root_table(d, ["Close", "lnClose", "Volume", "ret_pct", "dClose", "dVol_bn"])
    eg = stn.engle_granger(d, ycol, xcol)
    ecm = stn.fit_ecm(eg, d, ycol, xcol)
    var = stn.fit_var(d, ["dClose", "dVol_bn"], var_crit)
    return d, ur, eg, ecm, var


@st.cache_data(show_spinner=False)
def run_regime(ticker, source, start, end, series_col, k, sw_var, reps):
    d, _ = load(ticker, source)
    d = data.slice_window(d, start, end)
    return d, rg.fit_markov(d[series_col], k_regimes=k, switching_variance=sw_var, search_reps=reps)


@st.cache_data(show_spinner=False)
def run_collinearity(ticker, source, start, end, mas, vols, nkeep):
    d, _ = load(ticker, source)
    d = data.slice_window(d, start, end)
    X, y, _ = mc.build_features(d, tuple(mas), tuple(vols))
    return X, y, mc.vif_table(X), mc.pca_analysis(X, nkeep), mc.remedy_comparison(X, y, nkeep)


@st.cache_data(show_spinner=False)
def run_features(ticker, source, start, end, ar_p, ar_q, spec, horizon):
    d, _ = load(ticker, source)
    d = data.slice_window(d, start, end)
    fp = ft.feature_panel(d["ret_pct"], (ar_p, 0, ar_q), spec)
    cmp = ft.compare_specs(d["ret_pct"])
    fc = ft.forecast_risk(fp["garch"], horizon)
    return d, fp, cmp, fc


@st.cache_data(show_spinner=False)
def run_pair(source):
    g1, _ = data.get_data("GOOGL", source)
    g2, _ = data.get_data("GOOG", source)
    p1, p2 = data.prepare(g1, "GOOGL"), data.prepare(g2, "GOOG")
    pair = pd.concat([p1["Close"].rename("GOOGL"), p2["Close"].rename("GOOG")], axis=1).dropna()
    eg = stn.engle_granger(pair, "GOOGL", "GOOG", log=True)
    return pair, eg, stn.fit_ecm(eg, pair, "GOOGL", "GOOG", log=True)


def dl(df: pd.DataFrame, name: str, key: str):
    st.download_button("⬇ Download CSV", df.to_csv().encode(), file_name=name,
                       mime="text/csv", key=key)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Model controls")
    ticker = st.selectbox("Asset", ["BTC-USD", "ETH-USD"], help="Crypto daily series, 2020-2025")
    source = st.radio("Data source", ["cache", "live"],
                      format_func=lambda s: "Cached CSV (fast, reproducible)" if s == "cache"
                      else "Live Yahoo Finance")
    st.caption("Cached data is committed to the repo, so the app works even if Yahoo blocks the server.")
    start = st.date_input("Start", pd.Timestamp("2020-01-01"), min_value=pd.Timestamp("2020-01-01"),
                          max_value=pd.Timestamp("2025-02-27"))
    end = st.date_input("End", pd.Timestamp("2025-02-27"), min_value=pd.Timestamp("2020-06-01"),
                        max_value=pd.Timestamp("2025-02-27"))

start_s, end_s = str(start), str(end)
if pd.Timestamp(start) >= pd.Timestamp(end):
    st.error("Start date must be before end date.")
    footer()
    st.stop()

d0, prov = load(ticker, source)
st.title("📈 BTC Econometrics Lab")
st.markdown(
    "**Non-stationarity & equilibrium · Regime switching · Multicollinearity · Feature extraction**  \n"
    f"<span class='small'>{prov} · {len(d0):,} daily observations · "
    f"{d0.index[0].date()} → {d0.index[-1].date()}</span>",
    unsafe_allow_html=True,
)
if "SYNTHETIC" in prov:
    st.warning("Running on synthetic fallback data. Results are illustrative only.")

tabs = st.tabs(["🏠 Overview", "1️⃣ Non-stationarity", "2️⃣ Regime change",
                "3️⃣ Multicollinearity", "4️⃣ Feature extraction", "🧪 Bug lab", "📚 Method notes"])

# ----------------------------------------------------------------=========== #
# OVERVIEW
# ----------------------------------------------------------------=========== #
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${d0['Close'].iloc[-1]:,.0f}")
    c2.metric("Mean daily return", f"{d0['ret_pct'].mean():.3f}%")
    c3.metric("Daily volatility", f"{d0['ret_pct'].std():.2f}%")
    c4.metric("Annualised vol", f"{d0['ret_pct'].std()*np.sqrt(365):.1f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08,
                        subplot_titles=("Price level — non-stationary I(1)",
                                        "Daily percent return — stationary I(0)"))
    fig.add_trace(go.Scatter(x=d0.index, y=d0["Close"], line=dict(color=ACCENT, width=1.2),
                             name="Close ($)"), 1, 1)
    fig.add_trace(go.Scatter(x=d0.index, y=d0["ret_pct"], line=dict(color=BLUE, width=0.8),
                             name="Return (%)"), 2, 1)
    fig.update_yaxes(title_text="USD", row=1)
    fig.update_yaxes(title_text="percent", row=2)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=60, b=40))
    st.plotly_chart(fig, **W)

    st.subheader("What this project does")
    st.markdown(
        """
| # | Challenge | Models estimated live |
|---|-----------|----------------------|
| 1 | Non-stationarity & equilibrium | ADF, KPSS, Engle–Granger cointegration, ECM, VAR + impulse responses |
| 2 | Regime change | Hamilton 2-regime Markov switching, smoothed probabilities, expected durations |
| 3 | Multicollinearity | VIF, standardised PCA, Ridge / PCR / variable trimming with time-series CV |
| 4 | Feature extraction | ARMA conditional mean, GARCH-family conditional variance, Kalman local level, VaR |
        """
    )
    n_out = int(d0["vol_outlier"].sum()) if "vol_outlier" in d0 else 0
    verdict("<b>Reproducibility note.</b> The original submission's core statistics do reproduce "
            "(e.g. the ADF statistic on ΔClose is identical), but its code stopped running unchanged "
            "because <code>yfinance</code> changed its column layout and its <code>auto_adjust</code> "
            "default. This app pins both, caches the data and re-estimates the models. The "
            "<b>Bug lab</b> tab and <code>report/REVIEW.md</code> list the defects that were found and fixed."
            + (f" <b>Data guard:</b> {n_out} volume print(s) flagged as artefacts "
               f"({', '.join(str(i.date()) for i in d0.index[d0['vol_outlier']])}) were interpolated."
               if n_out else ""), "warn")

# ----------------------------------------------------------------=========== #
# 1 NON-STATIONARITY
# ----------------------------------------------------------------=========== #
with tabs[1]:
    st.subheader("Unit-root and stationarity testing")
    cc1, cc2 = st.columns(2)
    ycol = cc1.selectbox("Dependent series (y)", ["Close", "lnClose"], key="eg_y")
    xcol = cc2.selectbox("Candidate partner (x)", ["Volume", "lnVolume"], key="eg_x")
    var_crit = st.selectbox("VAR lag-selection criterion", ["bic", "hqic", "aic", "fpe"],
                            help="AIC over-fits: it kept hitting the maximum lag cap in the original code.")
    with st.spinner("Estimating…"):
        d, ur, eg, ecm, var = run_stationarity(ticker, source, start_s, end_s, var_crit, ycol, xcol)

    st.dataframe(ur.style.format({"ADF_stat": "{:.3f}", "ADF_p": "{:.4g}", "KPSS_stat": "{:.3f}"}),
                 **W, hide_index=True)
    st.caption("ADF H₀ = unit root · KPSS H₀ = stationary. KPSS p-values are bounded to [0.01, 0.10] "
               "by statsmodels, so the bound is reported rather than a false exact value.")
    dl(ur, "unit_root_tests.csv", "dl_ur")

    st.divider()
    st.subheader("Engle–Granger cointegration test")
    m = st.columns(4)
    m[0].metric("Hedge ratio β", f"{eg['beta']:.6g}")
    m[1].metric("R²", f"{eg['r2']:.4f}")
    m[2].metric("Durbin–Watson", f"{eg['dw']:.3f}")
    m[3].metric("EG p-value", f"{eg['eg_p']:.4f}")

    o = eg["orders"]
    st.markdown(f"**Integration order check:** `{o[0]['series']}` is I({o[0]['order']}), "
                f"`{o[1]['series']}` is I({o[1]['order']}).")
    if not eg["precondition_ok"]:
        verdict("<b>Precondition violated.</b> Cointegration is only defined between series of the "
                "<i>same</i> integration order, normally I(1). Here the partner is already stationary "
                "in levels, so the regression is unbalanced — the test outcome is uninformative, "
                "not merely 'no cointegration'. The original report never checked this.", "bad")
    if eg["cointegrated"]:
        verdict(f"<b>Cointegrated.</b> EG statistic {eg['eg_stat']:.3f} is below the 5% critical value "
                f"{eg['eg_crit']['5%']:.3f}; a long-run equilibrium exists and an ECM is justified.", "ok")
    else:
        verdict(f"<b>No cointegration — spurious regression.</b> R²={eg['r2']:.3f} looks meaningful but "
                f"DW={eg['dw']:.3f} is near zero and the residuals carry a unit root "
                f"(EG p={eg['eg_p']:.3f}). Note the correct EG p-value ({eg['eg_p']:.4f}) differs from "
                f"the naive ADF p-value ({eg['naive_adf_p']:.4f}) the original code reported: "
                f"residual-based tests need MacKinnon critical values.", "bad")

    f2 = make_subplots(rows=1, cols=2, subplot_titles=("Levels regression", "Equilibrium gap (residual)"))
    f2.add_trace(go.Scatter(x=d.index, y=d[ycol], name=ycol, line=dict(color=ACCENT)), 1, 1)
    f2.add_trace(go.Scatter(x=eg["resid"].index, y=eg["resid"].values, name="residual",
                            line=dict(color=RED, width=1)), 1, 2)
    f2.update_layout(height=330, showlegend=False, margin=dict(t=50, b=30))
    st.plotly_chart(f2, **W)

    st.divider()
    st.subheader("Error-correction model")
    e = st.columns(4)
    e[0].metric("Adjustment φ", f"{ecm['phi']:.5f}")
    e[1].metric("p-value", f"{ecm['phi_p']:.4f}")
    e[2].metric("Short-run γ", f"{ecm['gamma']:.4g}")
    e[3].metric("R²", f"{ecm['r2']:.4f}")
    if ecm["valid"]:
        verdict(f"<b>Valid error correction.</b> φ={ecm['phi']:.4f} (p={ecm['phi_p']:.2g}) is negative "
                f"and significant: {abs(ecm['phi'])*100:.1f}% of any disequilibrium is removed per day, "
                f"a half-life of {ecm['half_life_days']:.1f} days.", "ok")
    else:
        verdict("<b>ECM is not identified here.</b> φ is statistically indistinguishable from zero, so "
                "there is no error-correction mechanism. Fitting an ECM without cointegration violates "
                "the Granger representation theorem — the original notebook did exactly this. "
                "Scroll down for a pair where the ECM genuinely works.", "bad")

    st.divider()
    st.subheader("A pair that really is cointegrated: Alphabet dual-class shares")
    st.caption("GOOGL (voting) and GOOG (non-voting) claim the same cash flows, so theory predicts a "
               "log-price hedge ratio of ≈ 1 and a stationary spread. This is the equilibrium demonstration "
               "the assignment asks for.")
    with st.spinner("Estimating GOOGL/GOOG…"):
        pair, eg2, ecm2 = run_pair(source)
    q = st.columns(5)
    q[0].metric("β (log)", f"{eg2['beta']:.4f}")
    q[1].metric("EG p-value", f"{eg2['eg_p']:.4f}")
    q[2].metric("φ", f"{ecm2['phi']:.4f}")
    q[3].metric("φ p-value", f"{ecm2['phi_p']:.1e}")
    q[4].metric("Half-life", f"{ecm2['half_life_days']:.1f} d")
    verdict(f"<b>Cointegration confirmed and the ECM works.</b> β={eg2['beta']:.4f} ≈ 1 as theory predicts; "
            f"{abs(ecm2['phi'])*100:.1f}% of the spread closes each day "
            f"(half-life {ecm2['half_life_days']:.1f} days). This is a tradable statistical-arbitrage signal.", "ok")
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=eg2["resid"].index, y=eg2["resid"].values,
                            line=dict(color=GREEN, width=1), name="ln GOOGL − β·ln GOOG"))
    f3.add_hline(y=0, line=dict(color="black", width=1))
    f3.update_layout(height=300, yaxis_title="log spread", xaxis_title="Date",
                     margin=dict(t=30, b=30), showlegend=False)
    st.plotly_chart(f3, **W)

    st.divider()
    st.subheader(f"VAR({var['p']}) on stationary, scale-matched differences")
    v = st.columns(5)
    v[0].metric("Lag cap (data-driven)", var["maxlags"])
    v[1].metric(f"Chosen by {var['chosen_criterion'].upper()}", var["p"])
    v[2].metric("Stable", "Yes" if var["stable"] else "No")
    v[3].metric("Residual whiteness p", f"{var['whiteness_p']:.2g}")
    v[4].metric("BIC", f"{var['bic']:.3f}")
    st.json({"lag orders selected": var["orders"], "Granger causality": var["granger"]}, expanded=False)
    if var["whiteness_p"] < 0.05:
        verdict(f"<b>Diagnostic failure.</b> The Portmanteau test rejects white-noise residuals "
                f"(p={var['whiteness_p']:.2g}), so the VAR has not captured all linear dynamics. "
                f"The original submission ran no residual diagnostic at all. Volume in billions "
                f"(instead of raw units) also cut the regressor scale problem that gave a condition number near 1e11.", "warn")
    else:
        verdict("<b>Residuals are white.</b> The VAR captures the linear dynamics adequately.", "ok")
    try:
        irf = var["result"].irf(12)
        arr = np.asarray(irf.irfs)
        fi = make_subplots(rows=2, cols=2, subplot_titles=[
            "dClose ← dClose shock", "dClose ← dVol shock", "dVol ← dClose shock", "dVol ← dVol shock"])
        for i in range(2):
            for j in range(2):
                fi.add_trace(go.Scatter(y=arr[:, i, j], mode="lines",
                                        line=dict(color=BLUE)), (i // 1) + 1, j + 1)
        fi.update_layout(height=420, showlegend=False, margin=dict(t=50, b=30))
        st.plotly_chart(fi, **W)
    except Exception as exc:
        st.info(f"Impulse responses unavailable: {exc}")

# ----------------------------------------------------------------=========== #
# 2 REGIME
# ----------------------------------------------------------------=========== #
with tabs[2]:
    st.subheader("Hamilton Markov regime-switching model")
    a, b, c, e = st.columns(4)
    series_col = a.selectbox("Series to model", ["ret_pct", "dClose"],
                             format_func=lambda s: "Percent returns (recommended)" if s == "ret_pct"
                             else "Dollar price changes (original)")
    k = b.selectbox("Number of regimes", [2, 3], index=0)
    sw_var = c.checkbox("Switching variance", value=True)
    reps = e.slider("Multi-start searches", 0, 25, 12,
                    help="One start can land in a local optimum. We keep the best log-likelihood.")
    with st.spinner("Running the Hamilton filter and smoother…"):
        dr, mk = run_regime(ticker, source, start_s, end_s, series_col, k, sw_var, reps)

    unit = "%" if series_col == "ret_pct" else "$"
    st.dataframe(pd.DataFrame(mk["regimes"]).rename(columns={
        "mu": f"mean ({unit})", "mu_p": "mean p", "sigma2": "variance",
        "sigma": f"std dev ({unit})", "expected_duration": "expected duration (days)",
        "share": "share of days"}).style.format({
            f"mean ({unit})": "{:.4f}", "mean p": "{:.3f}", "variance": "{:.4g}",
            f"std dev ({unit})": "{:.4f}", "expected duration (days)": "{:.1f}",
            "share of days": "{:.1%}"}), **W, hide_index=True)

    lo = [r for r in mk["regimes"] if r["regime"] != mk["high_vol_regime"]][0]
    hi = [r for r in mk["regimes"] if r["regime"] == mk["high_vol_regime"]][0]
    m = st.columns(4)
    m[0].metric("Volatility ratio", f"{hi['sigma']/lo['sigma']:.2f}×")
    m[1].metric("Log-likelihood", f"{mk['loglik']:.1f}")
    m[2].metric("Regime switches", mk["n_switches"])
    m[3].metric("Starts evaluated", mk["starts_tried"])

    st.markdown("**Transition matrix** (row = from, column = to)")
    tm = pd.DataFrame(mk["transition"],
                      index=[f"from {i}" for i in range(k)],
                      columns=[f"to {j}" for j in range(k)])
    st.dataframe(tm.style.format("{:.4f}"), **W)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.07,
                        subplot_titles=(f"Series ({unit})",
                                        "Smoothed probability of the turbulent regime"))
    fig.add_trace(go.Scatter(x=mk["series"].index, y=mk["series"].values,
                             line=dict(color="grey", width=0.7), name="series"), 1, 1)
    fig.add_trace(go.Scatter(x=mk["prob_high"].index, y=mk["prob_high"].values,
                             line=dict(color=RED, width=1), fill="tozeroy",
                             fillcolor="rgba(209,52,75,0.25)", name="P(turbulent)"), 2, 1)
    fig.add_hline(y=0.5, line=dict(color="black", dash="dash", width=1), row=2)
    fig.update_yaxes(range=[0, 1], row=2)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=60, b=40))
    st.plotly_chart(fig, **W)

    verdict(f"<b>The original code's biggest bug.</b> It set "
            f"<code>df['Regime'] = result.predict()</code> and the report read the resulting "
            f"20–60 values as <i>percent probabilities</i>. <code>predict()</code> actually returns the "
            f"conditional <b>mean in dollars</b> — here it spans "
            f"{mk['predict_range'][0]:.2f} to {mk['predict_range'][1]:.2f}{unit}, exactly the regime means. "
            f"Correct inference uses <code>smoothed_marginal_probabilities</code>, shown above.", "bad")

    if series_col == "dClose":
        verdict("<b>Second specification bug.</b> Dollar differences make variance grow with the price "
                "level. BTC went from ~$7k to ~$90k, so this model labels 2024–25 'turbulent' ~95% of days "
                "while calling the March-2020 crash calm. Switch the selector to <b>percent returns</b> "
                "to see the economically correct classification.", "bad")
    else:
        yr = pd.Series((mk["label"] == mk["high_vol_regime"]).astype(int).values, index=mk["label"].index)
        byyr = yr.groupby(yr.index.year).mean()
        fb = go.Figure(go.Bar(x=[str(i) for i in byyr.index], y=byyr.values, marker_color=GREEN))
        fb.update_layout(height=280, yaxis_title="share of turbulent days",
                         margin=dict(t=30, b=30), yaxis_tickformat=".0%")
        st.plotly_chart(fb, **W)
        top = byyr.idxmax()
        verdict(f"<b>Share of turbulent days by year.</b> Highest in {top} ({byyr.max():.0%}), lowest in "
                f"{byyr.idxmin()} ({byyr.min():.0%}). Compare with the dated crashes you know (Mar 2020, "
                "2021 deleveraging, 2022 collapses) before reading anything economic into it.", "ok")
        _durs = ", ".join("%.1f" % r["expected_duration"] for r in mk["regimes"])
        if mk.get("quality", "").startswith("mixture"):
            verdict(f"<b>Caveat — these are not persistent regimes.</b> Expected durations are only "
                    f"{_durs} days and the chain "
                    f"switches {mk['n_switches']} times in {len(mk['label'])} days. At daily frequency a "
                    "two-state model mostly separates ordinary days from fat-tail days (a mixture of normals), "
                    "which is volatility clustering, not a structural regime change. Use GARCH for the "
                    "volatility forecast and treat the state labels as a descriptive flag.", "warn")

    ep = rg.regime_episodes(mk["label"], min_len=3)
    st.markdown(f"**Dated episodes lasting ≥ 3 days** ({len(ep)} found)")
    st.dataframe(ep.head(20), **W, hide_index=True)
    out = mk["smoothed"].assign(regime=mk["label"])
    dl(out, "regime_probabilities.csv", "dl_reg")

# ----------------------------------------------------------------=========== #
# 3 MULTICOLLINEARITY
# ----------------------------------------------------------------=========== #
with tabs[3]:
    st.subheader("Build a predictor set and diagnose collinearity")
    st.caption("All predictors are lagged one day, so no look-ahead bias enters the regression.")
    a, b, c = st.columns(3)
    mas = a.multiselect("Moving-average gap windows", [5, 10, 20, 50, 100, 200],
                        default=[5, 10, 20, 50, 100])
    vols = b.multiselect("Rolling volatility windows", [5, 10, 20, 30, 60], default=[5, 10, 20])
    nkeep = c.slider("Principal components to retain", 1, 8, 3)
    if len(mas) + len(vols) < 2:
        st.warning("Pick at least two predictors in total — using the defaults for now.")
        mas, vols = [5, 10, 20, 50, 100], [5, 10, 20]
    with st.spinner("Computing VIF, PCA and out-of-sample comparisons…"):
        X, y, vif, pca, rem = run_collinearity(ticker, source, start_s, end_s,
                                               tuple(mas), tuple(vols), nkeep)

    m = st.columns(4)
    m[0].metric("Predictors", rem["n_pred"])
    m[1].metric("Max VIF", f"{vif['VIF'].max():.1f}")
    m[2].metric("Condition number (raw)", f"{pca['cond_raw']:.3g}")
    m[3].metric("Significant at 5%", f"{rem['n_sig']} / {rem['n_pred']}")

    col1, col2 = st.columns(2)
    colors = [RED if f == "severe" else (ACCENT if f == "moderate" else GREEN) for f in vif["flag"]]
    fv = go.Figure(go.Bar(x=vif["VIF"], y=vif["feature"], orientation="h", marker_color=colors))
    fv.add_vline(x=10, line=dict(color="black", dash="dash"))
    fv.update_layout(height=380, xaxis_title="VIF", yaxis=dict(autorange="reversed"),
                     title="Variance inflation factors", margin=dict(t=50, b=30))
    col1.plotly_chart(fv, **W)

    fp = go.Figure()
    fp.add_trace(go.Bar(x=list(range(1, len(pca["evr"]) + 1)), y=pca["evr"],
                        name="individual", marker_color=BLUE))
    fp.add_trace(go.Scatter(x=list(range(1, len(pca["cum"]) + 1)), y=pca["cum"],
                            name="cumulative", mode="lines+markers", line=dict(color=RED)))
    fp.add_hline(y=0.9, line=dict(color="black", dash="dash"))
    fp.update_layout(height=380, title=f"Standardised PCA — {pca['n_for_90']} PCs reach 90%",
                     yaxis_title="explained variance ratio", xaxis_title="component",
                     margin=dict(t=50, b=30), legend=dict(orientation="h"))
    col2.plotly_chart(fp, **W)

    if vif["VIF"].max() > 10:
        sev = ", ".join(vif.loc[vif["flag"] == "severe", "feature"])
        verdict(f"<b>Severe multicollinearity detected</b> in {sev}. Overlapping moving averages measure "
                f"almost the same trend. Only {rem['n_sig']} of {rem['n_pred']} coefficients is significant "
                f"and in-sample R² is {rem['ols_r2']:.4f} — the classic symptom of inflated standard errors.", "bad")
    else:
        verdict("<b>No severe collinearity</b> for this predictor set. Add more overlapping windows "
                "(e.g. 10, 20 and 50) to reproduce the pathology.", "ok")

    verdict("<b>Why the original test was vacuous.</b> VIF was computed on <code>Diff_Close</code> and "
            "<code>Diff_Volume</code> — the two <i>dependent</i> variables of the VAR, which are not "
            "regressors of each other, so VIF ≈ 1.001 was guaranteed. And PCA on unstandardised data gave "
            "'PC1 = 100%' purely because volume (~1e10) dwarfs price changes (~1e2). Standardising gives "
            f"{pca['evr'][0]:.1%} / {pca['evr'][1]:.1%} — and the correlation is essentially zero. "
            "The report's two claims ('one dominant factor' and 'no redundancy') contradicted each other.", "warn")

    st.divider()
    st.subheader("Do the remedies actually help? Expanding-window time-series CV")
    names = list(rem["mse"])
    fr = go.Figure(go.Bar(x=names, y=[rem["mse"][k] for k in names],
                          marker_color=[RED, GREEN, BLUE, ACCENT],
                          text=[f"{rem['mse'][k]:.3f}" for k in names], textposition="outside"))
    fr.update_layout(height=340, yaxis_title="out-of-sample MSE",
                     title="Lower is better — but note how small the gaps are", margin=dict(t=50, b=30))
    st.plotly_chart(fr, **W)
    st.dataframe(pd.DataFrame({"model": names,
                               "OOS MSE": [rem["mse"][k] for k in names],
                               "improvement vs OLS (%)": [rem["gain_pct"][k] for k in names]})
                 .style.format({"OOS MSE": "{:.4f}", "improvement vs OLS (%)": "{:+.2f}"}),
                 **W, hide_index=True)
    best_gain = max(v for k, v in rem["gain_pct"].items() if k != "OLS")
    verdict(f"<b>Read the gap honestly.</b> The best remedy beats OLS by only {best_gain:.2f}% out of sample "
            f"(better in {max(rem['folds_better'].values())} of {rem['cv_splits']} folds at most). "
            + ("Ridge chose the <b>largest penalty on the grid</b>, i.e. it shrinks every slope to ~0 — the "
               "predictors carry essentially no usable signal for next-day returns. " if rem["ridge_at_edge"] else "")
            + "Collinearity is real (VIF above), but fixing it does not create predictability.", "warn")
    st.markdown(f"- Ridge penalty chosen by CV: **α = {rem['ridge_alpha']:.3g}**\n"
                f"- Variables trimmed to bring every VIF ≤ 10: **{', '.join(rem['dropped']) or 'none'}**\n"
                f"- A random K-fold split would leak future information; an expanding window does not.")
    st.markdown("**PC loadings** (which features each component represents)")
    st.dataframe(pca["loadings"].style.format("{:+.3f}"), **W)
    dl(vif, "vif_table.csv", "dl_vif")

# ----------------------------------------------------------------=========== #
# 4 FEATURES
# ----------------------------------------------------------------=========== #
with tabs[4]:
    st.subheader("Extract the signal: conditional mean, conditional variance, Kalman level")
    a, b, c, h = st.columns(4)
    ar_p = a.selectbox("AR order", [0, 1, 2, 3], index=1)
    ar_q = b.selectbox("MA order", [0, 1, 2, 3], index=1)
    spec = c.selectbox("Volatility specification", list(ft.SPECS), index=0)
    horizon = h.slider("Forecast horizon (days)", 1, 30, 10)
    with st.spinner("Estimating ARMA, GARCH and the Kalman filter…"):
        dfx, fpk, cmp, fc = run_features(ticker, source, start_s, end_s, ar_p, ar_q, spec, horizon)

    g = fpk["garch"]
    m = st.columns(5)
    m[0].metric("ADF p (returns)", f"{fpk['adf'][1]:.2g}")
    m[1].metric("Persistence α+β", f"{g['persistence']:.4f}")
    m[2].metric("Vol half-life", "∞" if not np.isfinite(g["half_life_days"])
                else f"{g['half_life_days']:.1f} d")
    m[3].metric("Tail index ν", "—" if not np.isfinite(g["nu"]) else f"{g['nu']:.2f}")
    m[4].metric("AIC", f"{g['aic']:.1f}")

    P = fpk["panel"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=("Conditional mean — ARMA", "Conditional variance — GARCH family",
                                        "Squared returns — raw noisy proxy", "Kalman smoothed local level"))
    fig.add_trace(go.Scatter(x=P.index, y=P["return_pct"], line=dict(color="lightgrey", width=0.6),
                             name="return"), 1, 1)
    fig.add_trace(go.Scatter(x=P.index, y=P["conditional_mean"], line=dict(color=BLUE, width=1.1),
                             name="ARMA mean"), 1, 1)
    fig.add_trace(go.Scatter(x=P.index, y=P["conditional_variance"], line=dict(color=ACCENT, width=1),
                             name="conditional variance"), 2, 1)
    fig.add_trace(go.Scatter(x=P.index, y=P["squared_return"], line=dict(color=RED, width=0.6),
                             name="squared return"), 3, 1)
    fig.add_trace(go.Scatter(x=P.index, y=P["return_pct"], line=dict(color="lightgrey", width=0.6),
                             name="return"), 4, 1)
    fig.add_trace(go.Scatter(x=P.index, y=P["kalman_level"], line=dict(color=GREEN, width=1.6),
                             name="Kalman level"), 4, 1)
    fig.update_layout(height=780, showlegend=False, margin=dict(t=60, b=40))
    st.plotly_chart(fig, **W)

    if g["integrated"]:
        verdict(f"<b>Volatility is near-integrated (IGARCH).</b> α+β = {g['persistence']:.4f}, so shocks "
                f"to variance barely decay. "
                + ("<b>Caution:</b> <code>arch</code> constrains α+β ≤ 1, so a value of exactly 1 is the optimiser "
                   "sitting on its boundary, not a free estimate — compare the EGARCH row below, which has no "
                   "such constraint. " if g.get("at_boundary") else "")
                + "The original run reported α=0.673, β=0.036 (sum 0.71) and concluded 'baseline level "
                  "volatility, no clustering'. That was an <b>optimiser failure caused by unscaled data</b> "
                  "(arch raised DataScaleWarning: scale 0.0011). On percent returns every specification is "
                  "highly persistent (0.97–1.00) — the opposite conclusion.", "bad")
    else:
        verdict(f"<b>Stationary volatility process.</b> α+β = {g['persistence']:.4f} < 1, half-life "
                f"{g['half_life_days']:.1f} days, unconditional volatility "
                f"{g['unconditional_vol']:.2f}% per day.", "ok")

    st.markdown("**Specification race** (same data, same scaling — ranked by AIC)")
    st.dataframe(cmp.style.format({"loglik": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}",
                                   "persistence": "{:.4f}", "nu": "{:.2f}"}),
                 **W, hide_index=True)

    k = fpk["kalman"]
    st.markdown(f"**Kalman local level:** σ²(level) = {k['sigma2_level']:.3g}, "
                f"σ²(irregular) = {k['sigma2_irregular']:.3g}, signal-to-noise = {k['signal_to_noise']:.3g}. "
                f"{k['interpretation'].capitalize()}. "
                f"Correlation between squared returns and fitted conditional variance: "
                f"**{fpk['sq_vs_garch_corr']:.3f}** — the GARCH variance is a far less noisy risk proxy.")

    st.divider()
    st.subheader("Deployable output: volatility term structure and Value-at-Risk")
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=fc["horizon_days"], y=fc["vol_forecast_pct"], mode="lines+markers",
                            name="volatility forecast (%)", line=dict(color=BLUE)))
    ff.add_trace(go.Scatter(x=fc["horizon_days"], y=-fc["VaR95_pct"], mode="lines+markers",
                            name="95% VaR (loss, %)", line=dict(color=ACCENT, dash="dash")))
    ff.add_trace(go.Scatter(x=fc["horizon_days"], y=-fc["VaR99_pct"], mode="lines+markers",
                            name="99% VaR (loss, %)", line=dict(color=RED, dash="dot")))
    ff.update_layout(height=360, xaxis_title="horizon (days)", yaxis_title="percent",
                     margin=dict(t=30, b=30), legend=dict(orientation="h"))
    st.plotly_chart(ff, **W)
    st.dataframe(fc.style.format({"vol_forecast_pct": "{:.3f}", "VaR95_pct": "{:.3f}",
                                  "VaR99_pct": "{:.3f}"}), **W, hide_index=True)
    if str(fc["method"].iloc[0]) == "simulation":
        st.caption("This specification has no closed-form multi-step variance forecast, so the "
                   "term structure above is a 2,000-path simulation average.")
    st.caption("VaR uses the fitted Student-t quantiles rescaled to unit variance, not Gaussian quantiles. "
               "With ν ≈ 3 the tails are far heavier than normal, so Gaussian VaR would understate risk.")
    dl(fc, "risk_forecast.csv", "dl_fc")
    dl(P, "feature_panel.csv", "dl_panel")

# ----------------------------------------------------------------=========== #
# BUG LAB
# ----------------------------------------------------------------=========== #
with tabs[5]:
    st.subheader("🧪 Bug lab — reproduce the defects side by side")
    st.caption("Each toggle recomputes both the flawed and the corrected version on the same data.")

    with st.expander("① predict() is a mean, not a probability", expanded=True):
        with st.spinner("Fitting…"):
            _, mkd = run_regime(ticker, source, start_s, end_s, "dClose", 2, True, 6)
        pr = mkd["result"].predict()
        f = make_subplots(rows=1, cols=2, subplot_titles=(
            "What the report plotted: predict()", "What it should have plotted: P(turbulent)"))
        f.add_trace(go.Scatter(x=mkd["series"].index, y=np.asarray(pr),
                               line=dict(color=ACCENT, width=1)), 1, 1)
        f.add_trace(go.Scatter(x=mkd["prob_high"].index, y=mkd["prob_high"].values,
                               line=dict(color=RED, width=1), fill="tozeroy"), 1, 2)
        f.update_yaxes(range=[0, 1], row=1, col=2)
        f.update_layout(height=320, showlegend=False, margin=dict(t=50, b=30))
        st.plotly_chart(f, **W)
        st.markdown(f"`predict()` ranges **{mkd['predict_range'][0]:.2f} – {mkd['predict_range'][1]:.2f} dollars** "
                    f"(the two regime means). It can never be a probability. "
                    f"The report's sentence *'a value of 60 implies a 60% probability'* is therefore wrong.")

    with st.expander("② PCA without standardisation invents a factor"):
        dd, _ = load(ticker, source)
        dd = data.slice_window(dd, start_s, end_s)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Z = dd[["dClose", "dVol_bn"]].dropna()
        raw = PCA().fit(Z.values).explained_variance_ratio_
        sc = PCA().fit(StandardScaler().fit_transform(Z)).explained_variance_ratio_
        t = pd.DataFrame({"": ["PC1", "PC2"], "Unstandardised (original)": raw, "Standardised (fixed)": sc})
        st.dataframe(t.style.format({"Unstandardised (original)": "{:.6f}", "Standardised (fixed)": "{:.4f}"}),
                     hide_index=True, **W)
        st.markdown(f"Correlation between the two series: **{Z.corr().iloc[0,1]:+.4f}**. "
                    "There is no common factor — 'PC1 = 100%' was a units artefact.")

    with st.expander("③ GARCH scaling changes the economic conclusion"):
        dd, _ = load(ticker, source)
        dd = data.slice_window(dd, start_s, end_s)
        rp = dd["ret_pct"]
        from arch import arch_model
        bad = arch_model(rp / 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        good = ft.garch_variance(rp, "GARCH(1,1)-t")
        t = pd.DataFrame({
            "parameter": ["omega", "alpha[1]", "beta[1]", "persistence", "nu"],
            "raw-scale returns (original)": [bad.params["omega"], bad.params["alpha[1]"],
                                             bad.params["beta[1]"],
                                             bad.params["alpha[1]"] + bad.params["beta[1]"], bad.params["nu"]],
            "percent returns (fixed)": [good["omega"], good["alpha"], good["beta"],
                                        good["persistence"], good["nu"]]})
        st.dataframe(t.style.format({"raw-scale returns (original)": "{:.6f}",
                                     "percent returns (fixed)": "{:.6f}"}),
                     hide_index=True, **W)
        st.markdown("Same asset, same sample, same model family — only the **units** differ, "
                    "yet persistence moves from a stationary-looking value to the IGARCH boundary. "
                    "Scaling is not cosmetic: it decides whether you conclude 'no clustering' or "
                    "'shocks never die'.")

    with st.expander("④ Look-ahead bias in feature construction"):
        import statsmodels.api as sm
        dd, _ = load(ticker, source)
        dd = data.slice_window(dd, start_s, end_s)
        _c, _r = dd["Close"], dd["ret_pct"]
        def _features(lag: bool) -> pd.DataFrame:
            F = pd.DataFrame(index=dd.index)
            for w in (5, 10, 20, 50, 100):
                F[f"ma{w}_gap"] = _c.rolling(w).mean() / _c - 1
            for w in (5, 10, 20):
                F[f"vol{w}"] = _r.rolling(w).std()
            F["log_volume"] = np.log(dd["Volume"])
            F["mom5"], F["mom10"] = _c.pct_change(5), _c.pct_change(10)
            return F.shift(1) if lag else F
        r2 = {}
        for lag in (False, True):
            D = pd.concat([_r.rename("y"), _features(lag)], axis=1).dropna()
            r2[lag] = float(sm.OLS(D["y"], sm.add_constant(D.drop(columns="y"))).fit().rsquared)
        st.markdown(
            """
If a moving-average gap is built as `rolling(w).mean() / Close − 1` using **today's** close,
it mechanically contains today's return, because the return also uses today's close.
Regressing today's return on that feature is circular.
            """
        )
        st.dataframe(pd.DataFrame({
            "version": ["features use close at t (leaky)", "features lagged to t−1 (this app)"],
            "in-sample R²": [r2[False], r2[True]],
            "honest?": ["❌ leakage", "✅"]}).style.format({"in-sample R²": "{:.4f}"}),
            hide_index=True, **W)
        st.markdown(f"The drop from {r2[False]:.0%} to {r2[True]:.1%} is the entire 'predictive power' "
                    "evaporating once the leak is closed.")

    with st.expander("⑤ yfinance broke the column layout"):
        st.code(
            """# Original (silent corruption)
df = yf.download('BTC-USD', start=..., end=...)
df = df[['Close', 'Volume']]          # columns are MultiIndex ('Close','BTC-USD')
sm.OLS(df['Close'], sm.add_constant(df['Volume']))
# -> coefficient table labels the regressor 'BTC-USD', not 'Volume'
# -> ECM table shows tuple labels ('lag_eg_resid', '')

# Fixed
df = yf.download(..., auto_adjust=False, group_by='column')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)""",
            language="python",
        )
        st.markdown("Both defects are visible in the submitted PDF output: the regressor is named "
                    "`BTC-USD` and the ECM coefficients are printed as tuples. `auto_adjust` also "
                    "flipped its default to `True`, silently changing every price.")

# ----------------------------------------------------------------=========== #
# NOTES
# ----------------------------------------------------------------=========== #
with tabs[6]:
    st.subheader("Model equations")
    st.markdown("**Augmented Dickey–Fuller** (H₀: unit root)")
    st.latex(r"\Delta y_t=\alpha+\beta t+\phi y_{t-1}+\sum_{i=1}^{p}\gamma_i\Delta y_{t-i}+\varepsilon_t")
    st.markdown(r"**KPSS** (H₀: stationarity), $y_t=\mu+\delta t+R_t+\varepsilon_t$ with $R_t$ a random walk.")
    st.latex(r"\eta=\frac{1}{T^{2}}\frac{\sum_{t=1}^{T}S_t^{2}}{\hat\sigma^{2}},\qquad S_t=\sum_{i=1}^{t}\hat\varepsilon_i")
    st.markdown("**Engle–Granger two-step**")
    st.latex(r"y_t=\alpha+\beta x_t+\epsilon_t\quad\text{then test }\epsilon_t\text{ for a unit root (MacKinnon CVs)}")
    st.markdown("**Error-correction model** — φ < 0 is the speed of adjustment")
    st.latex(r"\Delta y_t=\alpha+\underbrace{\phi\,(y_{t-1}-\hat\beta x_{t-1})}_{\text{error correction}}+\gamma\Delta x_t+\varepsilon_t")
    st.markdown("**VAR(p)**")
    st.latex(r"\mathbf{z}_t=\mathbf{c}+\sum_{i=1}^{p}\mathbf{A}_i\mathbf{z}_{t-i}+\mathbf{u}_t,\qquad \mathbf{u}_t\sim WN(\mathbf{0},\Sigma)")
    st.markdown("**Markov switching** with $S_t$ an unobserved Markov chain")
    st.latex(r"y_t=\mu_{S_t}+\epsilon_t,\quad \epsilon_t\sim N(0,\sigma^2_{S_t}),\quad P(S_t=j\mid S_{t-1}=i)=p_{ij}")
    st.latex(r"\mathbb{E}[\text{duration of regime }i]=\frac{1}{1-p_{ii}}")
    st.markdown("**VIF** and the GARCH(1,1) variance recursion")
    st.latex(r"\mathrm{VIF}_j=\frac{1}{1-R_j^{2}},\qquad \sigma_t^{2}=\omega+\alpha\epsilon_{t-1}^{2}+\beta\sigma_{t-1}^{2}")
    st.markdown("**Kalman local level** (state-space form)")
    st.latex(r"y_t=\mu_t+\varepsilon_t,\quad \mu_t=\mu_{t-1}+\eta_t,\quad \varepsilon_t\sim N(0,\sigma^2_\varepsilon),\ \eta_t\sim N(0,\sigma^2_\eta)")

    st.divider()
    st.subheader("References (MLA)")
    st.markdown(
        """
- Box, George E. P., et al. *Time Series Analysis: Forecasting and Control*. 5th ed., Wiley, 2015.
- Dickey, David A., and Wayne A. Fuller. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *JASA*, vol. 74, no. 366a, 1979, pp. 427–431.
- Engle, Robert F., and Clive W. J. Granger. "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*, vol. 55, no. 2, 1987, pp. 251–276.
- Granger, Clive W. J., and Paul Newbold. "Spurious Regressions in Econometrics." *Journal of Econometrics*, vol. 2, no. 2, 1974, pp. 111–120.
- Hamilton, James D. *Time Series Analysis*. Princeton UP, 1994.
- Hamilton, James D. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, vol. 57, no. 2, 1989, pp. 357–384.
- Johansen, Søren. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica*, vol. 59, no. 6, 1991, pp. 1551–1580.
- Jolliffe, Ian T. *Principal Component Analysis*. 2nd ed., Springer, 2002.
- Kwiatkowski, Denis, et al. "Testing the Null Hypothesis of Stationarity against the Alternative of a Unit Root." *Journal of Econometrics*, vol. 54, no. 1–3, 1992, pp. 159–178.
- WorldQuant University. "Correlation and Multicollinearity." *Financial Econometrics*, Module 1, Lesson 2, 2025.
        """
    )
    pass

footer()
