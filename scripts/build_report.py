"""Build assets/docs/Report.pdf (no code, numbers read from report/results.json)."""
from __future__ import annotations
import json, pathlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, CondPageBreak, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "report" / "results.json").read_text())
FIG = ROOT / "figures"
D = "/usr/share/fonts/truetype/dejavu/"
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(n, D + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="DV", fontSize=9, leading=12.5, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=B, fontName="DV-B", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#1e1b4b"))
H2 = ParagraphStyle("H2", parent=B, fontName="DV-B", fontSize=10.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#4338ca"))
T = ParagraphStyle("T", parent=B, fontName="DV-B", fontSize=22, leading=27, textColor=colors.HexColor("#1e1b4b"))
S = ParagraphStyle("S", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#374151"))
WARN = ParagraphStyle("W", parent=B, fontSize=8.3, leading=11.5, backColor=colors.HexColor("#fef2f2"), borderColor=colors.HexColor("#ef4444"),
                      borderWidth=0.8, borderPadding=6, textColor=colors.HexColor("#7f1d1d"))
EQ = ParagraphStyle("EQ", parent=B, alignment=1, fontName="DV-I", spaceBefore=2, spaceAfter=6)


def tbl(rows, widths=None, hl=None):
    t = Table([[Paragraph(str(c), S) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#c7d2fe")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(st)); return t


def img(name, w=15.5):
    from PIL import Image as PI
    iw, ih = PI.open(FIG / name).size
    return Image(str(FIG / name), width=w * cm, height=w * cm * ih / iw)


def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#6b7280"))
    c.drawString(2 * cm, 1.1 * cm, "BTC Econometrics Lab · MScFE 610 GWP 2 · Group 8665 · Educational use only — not financial, investment or professional advice.")
    c.drawRightString(19 * cm, 1.1 * cm, f"Page {d.page}"); c.restoreState()


ur = {r["series"]: r for r in R["unit_roots"]}
eg, ecm, var, eg2, ecm2 = R["eg_btc_volume"], R["ecm_btc_volume"], R["var"], R["eg_googl_goog"], R["ecm_googl_goog"]
mk, rem, ft = R["markov_returns"], R["remedies"], R["features"]
r0, r1 = mk["regimes"]; P = mk["transition"]; g = ft["garch_params"]; rf = R["risk_forecast"][0]
spec = {s["spec"]: s for s in ft["spec_table"]}
by = R["highvol_share_by_year"]["returns_spec"]
best_gain = max(v for k, v in rem["gain_pct"].items() if k != "OLS")

E = []
E += [Paragraph("BTC Econometrics Lab", T), Paragraph("Non-stationarity &amp; equilibrium · Regime switching · Multicollinearity · Feature extraction", B),
      Paragraph("MScFE 610 Financial Econometrics — Group Work Project 2 · Group 8665 · corrected release v2.1<br/>"
                "Developer: Ajeet Kumar · ajeetk095@gmail.com · Co-authors of the original coursework: Kow Kai Sheng, Faith Nyaboe Momanyi", S), Spacer(1, 6),
      Paragraph("<b>⚠ Important disclaimer.</b> This document is an educational and research artefact. Nothing in it is financial, investment, trading, legal, tax, "
                "medical or other professional advice, nor a recommendation to buy, sell, hold or trade any asset. Data are third-party and may be wrong; model outputs rest on "
                "simplified assumptions and a limited sample and can be wrong or misleading. Consult a qualified professional before any decision. The authors accept no liability.", WARN),
      Paragraph("Executive summary", H1),
      Paragraph(f"We analyse {R['n_obs']:,} daily BTC-USD observations ({R['window'][0]} to {R['window'][1]}). The review of the original submission found 20 defects; a second pass on "
                f"the corrected version found six more (see <i>REVIEW.md</i>). The headline results, all re-estimated from the committed data cache:", B),
      tbl([["Challenge", "Result"],
           ["1 · Non-stationarity", f"Price is I(1) (ADF p = {ur['Close']['ADF_p']:.2f}); returns are I(0) (p ≈ {ur['ret_pct']['ADF_p']:.0e}). Close vs volume is <b>not</b> cointegrated (EG p = {eg['eg_p']:.2f}); GOOGL/GOOG is (EG p = {eg2['eg_p']:.3f}, half-life {ecm2['half_life_days']:.1f} days)."],
           ["2 · Regime change", f"Two states: calm σ = {r0['sigma']:.2f}% (mean duration {r0['expected_duration']:.1f} d) and turbulent σ = {r1['sigma']:.2f}% ({r1['expected_duration']:.1f} d). Short-lived ⇒ a fat-tail mixture rather than structural regimes."],
           ["3 · Multicollinearity", f"VIF up to {max(v['VIF'] for v in R['vif']):.1f}; three components carry {R['pca']['evr'][0]+R['pca']['evr'][1]+R['pca']['evr'][2]:.0%} of predictor variance. Best remedy beats OLS by only {best_gain:.2f}% out of sample."],
           ["4 · Feature extraction", f"GARCH(1,1)-t: α = {g['alpha[1]']:.3f}, β = {g['beta[1]']:.3f}, ν = {g['nu']:.2f}; persistence on the optimiser bound ({ft['persistence']:.3f}); EGARCH-t (best AIC) {spec['EGARCH(1,1)-t']['persistence']:.3f}. One-day 95% / 99% VaR ≈ {-rf['VaR95_pct']:.1f}% / {-rf['VaR99_pct']:.1f}%."]],
          [3.6 * cm, 13.4 * cm]), CondPageBreak(9 * cm)]

# ---- Challenge 1
E += [Paragraph("1 · Non-stationarity and equilibrium", H1),
      Paragraph("<b>Description.</b> A series is stationary if its mean, variance and autocovariances do not depend on time. Prices of liquid assets typically contain a unit root; returns do not. "
                "We test H<sub>0</sub>: unit root (ADF) and H<sub>0</sub>: stationarity (KPSS) jointly.", B),
      Paragraph("Δy<sub>t</sub> = α + βt + φ y<sub>t−1</sub> + Σ γ<sub>i</sub> Δy<sub>t−i</sub> + ε<sub>t</sub>", EQ),
      tbl([["Series", "ADF stat", "ADF p", "KPSS", "Verdict"]] +
          [[k, f"{ur[k]['ADF_stat']:.2f}", f"{ur[k]['ADF_p']:.2g}", f"{ur[k]['KPSS_stat']:.2f} (p {ur[k]['KPSS_p']})", ur[k]["verdict"]] for k in ["Close", "lnClose", "Volume", "dClose", "ret_pct", "dVol_bn"]],
          [2.4 * cm, 2.2 * cm, 2.2 * cm, 3.6 * cm, 6.6 * cm]), Spacer(1, 4), img("01_stationarity.png", 11.5),
      Paragraph("<b>Diagnosis.</b> Close is I(1) while volume is stationary in levels, so the Engle–Granger precondition (both I(1)) fails: "
                f"the regression has R² = {eg['r2']:.2f} but a Durbin–Watson of {eg['dw']:.2f} — the signature of a spurious regression — and the residual test with MacKinnon values gives p = {eg['eg_p']:.2f}. "
                f"An error-correction term fitted anyway is insignificant (φ = {ecm['phi']:.4f}, p = {ecm['phi_p']:.2f}). A genuine equilibrium exists for the dual-class pair GOOGL/GOOG: "
                f"β = {eg2['beta']:.3f}, φ = {ecm2['phi']:.3f} (p ≈ {ecm2['phi_p']:.0e}). For the stationary differences, a VAR({var['p']}) chosen by BIC is stable; residual whiteness is rejected "
                f"(p = {var['whiteness_p']:.1g}) and ΔVolume Granger-causes ΔClose (p = {var['granger']['p']:.3f}) with negligible economic content.", B),
      Paragraph("<b>Damage / Directions.</b> Reading a spurious R² as a relationship, or an ECM without cointegration, invites false trading narratives. Use returns for modelling, test the integration order of "
                "<i>both</i> series, and reserve ECMs for pairs that pass MacKinnon-valued tests. <b>Data note:</b> one volume print (2021-02-26) was a 5.8× outlier that reverted next day; it is interpolated.", B),
      img("02_spurious.png", 13.5), CondPageBreak(9 * cm)]

# ---- Challenge 2
E += [Paragraph("2 · Detecting regime changes", H1),
      Paragraph("<b>Description.</b> A hidden Markov chain S<sub>t</sub> selects the mean and variance of returns (Hamilton):", B),
      Paragraph("y<sub>t</sub> = μ<sub>S<sub>t</sub></sub> + ε<sub>t</sub>,  ε<sub>t</sub> ~ N(0, σ²<sub>S<sub>t</sub></sub>),  P(S<sub>t</sub>=j | S<sub>t−1</sub>=i) = p<sub>ij</sub>,  E[duration<sub>i</sub>] = 1/(1 − p<sub>ii</sub>)", EQ),
      tbl([["Regime", "μ (%/day)", "σ (%/day)", "Share of days", "Expected duration"],
           ["0 calm", f"{r0['mu']:.3f}", f"{r0['sigma']:.2f}", f"{r0['share']:.0%}", f"{r0['expected_duration']:.1f} days"],
           ["1 turbulent", f"{r1['mu']:.3f}", f"{r1['sigma']:.2f}", f"{r1['share']:.0%}", f"{r1['expected_duration']:.1f} days"]], [3 * cm, 3 * cm, 3 * cm, 3.5 * cm, 4.5 * cm]),
      Paragraph(f"Transition matrix: p<sub>00</sub> = {P[0][0]:.3f}, p<sub>11</sub> = {P[1][1]:.3f}.", S), Spacer(1, 4), img("05_regime.png", 12.0),
      Paragraph("<b>Diagnosis.</b> Estimated on <i>dollar</i> differences the model labels 2021 turbulent on 100% of days and the March-2020 crash calm — the variance simply scales with the price level. "
                "On percent returns the turbulent share is " + ", ".join(f"{y}: {v:.0%}" for y, v in by.items()) + ". The original report also read <font face='DV-I'>predict()</font> "
                "(a dollar conditional mean) as a probability; regime inference must use the smoothed probabilities.", B),
      Paragraph(f"<b>Caveat.</b> With expected durations of {r0['expected_duration']:.1f} and {r1['expected_duration']:.1f} days and {mk['n_switches']} switches in {R['n_obs']:,} days, the two states mostly "
                "separate ordinary days from fat-tail days. That is volatility clustering, not a structural break; use GARCH for risk forecasts and treat the label as descriptive.", B),
      img("05b_regime_artefact.png", 8.5), CondPageBreak(9 * cm)]

# ---- Challenge 3
vif_top = sorted(R["vif"], key=lambda v: -v["VIF"])[:5]
E += [Paragraph("3 · Multicollinearity", H1),
      Paragraph("<b>Description.</b> Overlapping predictors inflate coefficient variances: VIF<sub>j</sub> = 1/(1 − R²<sub>j</sub>). All predictors are lagged one day, standardised before PCA, and models are compared "
                "with an expanding-window time-series cross-validation (a random split would leak the future).", B),
      tbl([["Predictor", "VIF", "Flag"]] + [[v["feature"], f"{v['VIF']:.1f}", v["flag"]] for v in vif_top], [5 * cm, 3 * cm, 4 * cm]), Spacer(1, 4), img("06_collinearity.png", 14.0),
      tbl([["Model", "OOS MSE", "Gain vs OLS (%)", "Folds better than OLS (of 5)"]] +
          [[k, f"{rem['mse'][k]:.4f}", f"{rem['gain_pct'][k]:+.2f}", str(rem['folds_better'][k])] for k in rem["mse"]], [4 * cm, 3.5 * cm, 4 * cm, 5.5 * cm]), Spacer(1, 4),
      Paragraph(f"<b>Diagnosis.</b> Only {rem['n_sig']} of {rem['n_pred']} coefficients is significant and in-sample R² is {rem['ols_r2']:.3f}. Standardised PCA needs {R['pca']['n_for_90']} components for 90% of the variance. "
                f"<b>Honest reading:</b> the best remedy improves on OLS by just {best_gain:.2f}%, and Ridge chooses the largest penalty on its grid (α = {rem['ridge_alpha']:.0e}), "
                "shrinking every slope towards zero — daily BTC returns are almost unforecastable from these predictors. Removing collinearity stabilises coefficients; it does not create predictive power.", B),
      Paragraph("The original VIF was computed on the two <i>dependent</i> variables (guaranteeing VIF ≈ 1), and PCA on unstandardised data returned “PC1 = 100%” purely because volume dwarfs price changes.", B),
      CondPageBreak(9 * cm)]

# ---- Challenge 4
E += [Paragraph("4 · Feature extraction and risk", H1),
      Paragraph("<b>Description.</b> Returns are compressed into four features: an ARMA conditional mean, a GARCH conditional variance, squared returns and a Kalman local level.", B),
      Paragraph("σ²<sub>t</sub> = ω + α ε²<sub>t−1</sub> + β σ²<sub>t−1</sub>,   VaR<sub>c</sub> = −(μ + z<sub>1−c</sub> σ),  z = t<sub>ν</sub>⁻¹(1−c)·√((ν−2)/ν)", EQ),
      tbl([["Specification", "AIC", "α+β", "ν"]] + [[s["spec"], f"{s['AIC']:.1f}", f"{s['persistence']:.3f}", "—" if s["nu"] != s["nu"] or s["nu"] is None else f"{s['nu']:.2f}"] for s in ft["spec_table"]],
          [6 * cm, 3.5 * cm, 3 * cm, 3 * cm]), Spacer(1, 4), img("08_features.png", 12.0),
      Paragraph(f"<b>Diagnosis.</b> Returns are stationary (ADF p ≈ {ft['adf_p']:.0e}). GARCH-t gives ω = {g['omega']:.3f}, α = {g['alpha[1]']:.3f}, β = {g['beta[1]']:.3f}, ν = {g['nu']:.2f}. "
                "The original run ignored arch's DataScaleWarning, got α = 0.673, β = 0.036 and concluded there was no clustering — an optimiser failure on unscaled data. "
                "Because <i>arch</i> constrains α+β ≤ 1, the exact 1.000 is the optimiser's bound; the unconstrained EGARCH-t (best AIC) gives "
                f"{spec['EGARCH(1,1)-t']['persistence']:.3f}. The Kalman signal-to-noise ratio ({ft['kalman']['signal_to_noise']:.0e}) shows no persistent drift.", B),
      img("09_risk.png", 9.5),
      Paragraph(f"<b>Damage.</b> One-day VaR from the fitted model is {-rf['VaR95_pct']:.1f}% (95%) and {-rf['VaR99_pct']:.1f}% (99%). A first version of the code divided the t-quantile by the scale factor and "
                "reported 10.8% / 20.7% — about 2.9 times too large; the corrected figures are shown here. At 99% a normal model (≈6.4%) understates the t-based tail; at 95% the normal figure (≈4.5%) is actually larger.", B),
      Paragraph("<b>Directions.</b> Prefer EGARCH-t, consider component/FIGARCH models, re-estimate on a rolling window, and back-test VaR (Kupiec, Christoffersen) — not yet done here.", B), CondPageBreak(9 * cm)]

# ---- Limitations + references
E += [Paragraph("Limitations and open items", H1),
      Paragraph("• Single asset, single five-year sample dominated by crypto-specific events. • No directional forecasting skill is claimed or shown. • VaR is not back-tested. "
                "• Live data (CoinGecko) are third-party and may be delayed; the app falls back to cached data and says so. • The GOOGL/GOOG gap shows an unexplained mid-2021 excursion worth a data check.", B),
      Paragraph("References (MLA)", H1)]
for ref in ["Box, George E. P., et al. <i>Time Series Analysis: Forecasting and Control</i>. 5th ed., Wiley, 2015.",
            "Dickey, David A., and Wayne A. Fuller. “Distribution of the Estimators for Autoregressive Time Series with a Unit Root.” <i>JASA</i>, vol. 74, no. 366a, 1979, pp. 427–431.",
            "Engle, Robert F., and Clive W. J. Granger. “Co-integration and Error Correction.” <i>Econometrica</i>, vol. 55, no. 2, 1987, pp. 251–276.",
            "Granger, Clive W. J., and Paul Newbold. “Spurious Regressions in Econometrics.” <i>Journal of Econometrics</i>, vol. 2, no. 2, 1974, pp. 111–120.",
            "Hamilton, James D. “A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.” <i>Econometrica</i>, vol. 57, no. 2, 1989, pp. 357–384.",
            "Kwiatkowski, Denis, et al. “Testing the Null Hypothesis of Stationarity against the Alternative of a Unit Root.” <i>Journal of Econometrics</i>, vol. 54, 1992, pp. 159–178.",
            "Bollerslev, Tim. “Generalized Autoregressive Conditional Heteroskedasticity.” <i>Journal of Econometrics</i>, vol. 31, no. 3, 1986, pp. 307–327.",
            "WorldQuant University. “Correlation and Multicollinearity.” <i>Financial Econometrics</i>, Module 1, Lesson 2, 2025."]:
    E.append(Paragraph(ref, S))

out = ROOT / "assets" / "docs" / "Report.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                  title="BTC Econometrics Lab — Report", author="Ajeet Kumar").build(E, onFirstPage=footer, onLaterPages=footer)
print("saved", out)
