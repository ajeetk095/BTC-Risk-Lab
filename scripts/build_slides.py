"""Generate the 15-slide project presentation (16:9)."""
from __future__ import annotations
import json, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "report" / "results.json").read_text())
FIG = ROOT / "figures"
OUT = ROOT / "assets" / "docs" / "Presentation.pptx"

NAVY = RGBColor(0x10, 0x1B, 0x2D)
NAVY2 = RGBColor(0x1B, 0x2A, 0x44)
ORANGE = RGBColor(0xF7, 0x93, 0x1A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x5A, 0x67, 0x78)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
GREEN = RGBColor(0x1E, 0x8E, 0x4E)
RED = RGBColor(0xC0, 0x2A, 0x42)
BLUE = RGBColor(0x2F, 0x6F, 0xB5)

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def rect(s, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def tb(s, x, y, w, h, text, size=16, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
       font="Calibri", space_after=6, anchor=MSO_ANCHOR.TOP, line_spacing=1.05):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = font
    return box


def bullets(s, x, y, w, h, items, size=15, color=NAVY, gap=8, bullet_color=ORANGE):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0); tf.margin_top = Emu(0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.line_spacing = 1.06
        r0 = p.add_run(); r0.text = "▸ "
        r0.font.size = Pt(size); r0.font.color.rgb = bullet_color; r0.font.bold = True
        r0.font.name = "Calibri"
        # allow (**bold lead**, rest) tuples
        if isinstance(it, tuple):
            lead, rest = it
            r1 = p.add_run(); r1.text = lead
            r1.font.size = Pt(size); r1.font.bold = True; r1.font.color.rgb = color
            r1.font.name = "Calibri"
            r2 = p.add_run(); r2.text = rest
            r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = "Calibri"
        else:
            r1 = p.add_run(); r1.text = it
            r1.font.size = Pt(size); r1.font.color.rgb = color; r1.font.name = "Calibri"
    return box


def header(s, num, title, sub=None):
    rect(s, 0, 0, W, Inches(1.02), NAVY)
    rect(s, 0, Inches(1.02), W, Inches(0.055), ORANGE)
    tb(s, Inches(0.55), Inches(0.16), Inches(10.6), Inches(0.45), title, 25, WHITE, True)
    if sub:
        tb(s, Inches(0.57), Inches(0.63), Inches(10.6), Inches(0.3), sub, 12.5,
           RGBColor(0xB9, 0xC6, 0xD6))
    # slide number chip
    rect(s, Inches(12.25), Inches(0.24), Inches(0.62), Inches(0.5), ORANGE)
    tb(s, Inches(12.25), Inches(0.31), Inches(0.62), Inches(0.36), f"{num:02d}", 17,
       NAVY, True, PP_ALIGN.CENTER)


def footer(s, text="MScFE 610 Financial Econometrics · GWP 2 · Group 8665"):
    tb(s, Inches(0.55), Inches(7.03), Inches(11.0), Inches(0.3), text, 9.5, GREY)


def callout(s, x, y, w, h, title, body, accent=ORANGE, bg=LIGHT, tsize=13.5, bsize=12.5):
    rect(s, x, y, w, h, bg)
    rect(s, x, y, Inches(0.06), h, accent)
    tb(s, x + Inches(0.22), y + Inches(0.12), w - Inches(0.4), Inches(0.28), title, tsize, accent, True)
    tb(s, x + Inches(0.22), y + Inches(0.46), w - Inches(0.42), h - Inches(0.55), body, bsize, NAVY)


def kpi(s, x, y, w, h, value, label, color=NAVY, bg=LIGHT, vsize=22, lsize=10.5):
    rect(s, x, y, w, h, bg)
    tb(s, x + Inches(0.12), y + Inches(0.13), w - Inches(0.24), Inches(0.42), value, vsize, color, True)
    tb(s, x + Inches(0.12), y + Inches(0.62), w - Inches(0.24), h - Inches(0.68), label, lsize, GREY)


def table(s, x, y, w, rows, col_w=None, size=11.5, header_size=11.5, row_h=0.32,
          head_h=0.36, highlight=None):
    nrow, ncol = len(rows), len(rows[0])
    shp = s.shapes.add_table(nrow, ncol, x, y, w, Inches(head_h + row_h * (nrow - 1)))
    t = shp.table
    if col_w:
        total = sum(col_w)
        for j, cw in enumerate(col_w):
            t.columns[j].width = Emu(int(w * cw / total))
    t.rows[0].height = Inches(head_h)
    for i in range(1, nrow):
        t.rows[i].height = Inches(row_h)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = Inches(0.09); c.margin_right = Inches(0.07)
            c.margin_top = Inches(0.03); c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = NAVY2
            elif highlight and i in highlight:
                c.fill.fore_color.rgb = RGBColor(0xFD, 0xF0, 0xDD)
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(header_size if i == 0 else size)
            r.font.bold = (i == 0)
            r.font.color.rgb = WHITE if i == 0 else NAVY
            r.font.name = "Calibri"
    return t


def pic(s, path, x, y, w=None, h=None):
    if w and h:
        return s.shapes.add_picture(str(path), x, y, width=w, height=h)
    if w:
        return s.shapes.add_picture(str(path), x, y, width=w)
    return s.shapes.add_picture(str(path), x, y, height=h)


def new(num, title, sub=None):
    s = prs.slides.add_slide(BLANK)
    header(s, num, title, sub); footer(s)
    return s


# ---------------------------------------------------------------- SLIDE 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, Inches(0.22), H, ORANGE)
tb(s, Inches(0.85), Inches(1.15), Inches(11.5), Inches(0.4),
   "WORLDQUANT UNIVERSITY  ·  MScFE 610 FINANCIAL ECONOMETRICS", 14, ORANGE, True)
tb(s, Inches(0.85), Inches(1.72), Inches(11.6), Inches(1.5),
   "BTC Econometrics Lab", 54, WHITE, True)
tb(s, Inches(0.85), Inches(2.85), Inches(11.6), Inches(1.0),
   "Non-stationarity & Equilibrium · Regime Switching ·\nMulticollinearity · Feature Extraction",
   22, RGBColor(0xC9, 0xD6, 0xE4))
rect(s, Inches(0.85), Inches(4.05), Inches(4.2), Inches(0.045), ORANGE)
tb(s, Inches(0.85), Inches(4.32), Inches(6.2), Inches(1.4),
   "Group Work Project 2  ·  Group 8665\nKow Kai Sheng (Malaysia)\nAjeet Kumar (India)\nFaith Nyaboe Momanyi (Kenya)",
   15, WHITE, space_after=4)
tb(s, Inches(7.6), Inches(4.32), Inches(4.9), Inches(1.4),
   "Review · Repair · Redeploy\n26 defects found and fixed\n1 live, free, interactive app\n15-slide briefing",
   15, ORANGE, True, space_after=4)
tb(s, Inches(0.85), Inches(6.35), Inches(11.6), Inches(0.6),
   "Asset: BTC-USD daily, 2020-01-01 → 2025-02-27 (1,884 observations) · "
   "Validation pair: GOOGL / GOOG · Source: Yahoo Finance",
   12.5, RGBColor(0x9F, 0xB2, 0xC8))
tb(s, Inches(0.85), Inches(6.85), Inches(11.6), Inches(0.5),
   "⚠ Educational and research use only — not financial, investment, trading or professional advice. "
   "Models and live data can be wrong. No liability is accepted for decisions based on this material.",
   10.5, RGBColor(0xFC, 0xA5, 0xA5))

# ---------------------------------------------------------------- SLIDE 2
s = new(2, "Executive summary — what the review found",
        "The modelling ambition was right; four results were reversed by implementation defects")
kpi(s, Inches(0.55), Inches(1.35), Inches(2.3), Inches(1.15), "20", "defects identified\nand fixed", ORANGE)
kpi(s, Inches(2.99), Inches(1.35), Inches(2.3), Inches(1.15), "4", "conclusions that\nchange direction", RED)
kpi(s, Inches(5.43), Inches(1.35), Inches(2.3), Inches(1.15), "0", "errors on a clean\nend-to-end run", GREEN)
kpi(s, Inches(7.87), Inches(1.35), Inches(2.3), Inches(1.15), "7", "interactive app tabs\nstress-tested", BLUE)
kpi(s, Inches(10.31), Inches(1.35), Inches(2.45), Inches(1.15), "100%", "free to run\nno API key", NAVY)

rows = [["#", "Defect", "What the report concluded", "What the corrected model shows"],
      ["1", "predict() read as a probability", "“60 means a 60% chance of Regime 1”",
       "predict() is a dollar mean (21–59); use smoothed probabilities"],
      ["2", "Regimes on dollar differences", "2021 turbulent, 2024 stable",
       "2024–25 flagged turbulent 95% of days — a price-level artefact"],
      ["3", "GARCH fitted on unscaled returns", "α+β = 0.71, “no volatility clustering”",
       "α+β = 1.00 (optimiser bound; EGARCH 0.99): shocks fade very slowly"],
      ["4", "VIF/PCA on the wrong variables", "“no multicollinearity” + “one dominant factor”",
       "Real predictor set: VIF up to 27.6; PCA artefact removed"]]
table(s, Inches(0.55), Inches(2.78), Inches(12.2), rows, col_w=[0.4, 2.5, 3.4, 4.4],
      size=11, row_h=0.62, head_h=0.4, highlight={1, 2, 3, 4})
callout(s, Inches(0.55), Inches(6.02), Inches(12.2), Inches(0.86),
        "BOTTOM LINE",
        "None of these are coding typos — each one flips an investment conclusion. All four are reproducible "
        "side-by-side in the app's “Bug lab” tab.", RED, RGBColor(0xFD, 0xEC, 0xEE), 12.5, 12)

# ---------------------------------------------------------------- SLIDE 3
s = new(3, "The brief, and how the project answers it",
        "Set D GWP2: pick 3 of 4 challenges (one must be non-stationarity or regime change)")
rows = [["Required task", "Where it is delivered", "Status"],
      ["Write model equations, define terms", "Method notes tab + notebook LaTeX (8 models)", "Met"],
      ["Import, structure and model the data", "src/data.py — flat columns, pinned auto_adjust, cached CSV", "Fixed"],
      ["Calibrate parameters", "MLE / OLS, multi-start search, CV-tuned ridge penalty", "Fixed"],
      ["Interpret parameters", "Every metric has an economic reading in-app", "Met"],
      ["Report in PDF, no code", "assets/docs/Report.pdf + report/REVIEW.md + 15-slide deck", "Met"],
      ["Executable notebook + HTML/PDF copy", "notebooks/*.ipynb executed, 0 errors + .html export", "Met"]]
table(s, Inches(0.55), Inches(1.4), Inches(7.5), rows, col_w=[2.6, 3.7, 0.8],
      size=11, row_h=0.44, head_h=0.38)

tb(s, Inches(8.35), Inches(1.4), Inches(4.4), Inches(0.35), "Four challenges covered", 16, NAVY, True)
bullets(s, Inches(8.35), Inches(1.85), Inches(4.4), Inches(2.4), [
    ("1 · Non-stationarity & equilibrium — ", "ADF, KPSS, Engle–Granger, ECM, VAR"),
    ("2 · Regime change — ", "2- and 3-state Markov switching"),
    ("3 · Multicollinearity — ", "VIF, PCA, Ridge, PCR"),
    ("4 · Feature extraction — ", "ARMA, GARCH family, Kalman"),
], size=13, gap=10)
callout(s, Inches(8.35), Inches(4.35), Inches(4.4), Inches(1.15),
        "RUBRIC GAPS TO CLOSE",
        "Report runs 30 pages against a 5–6 page limit; equations are pasted Unicode, not an equation "
        "editor; “Demonstration” sections defer to the code instead of showing output.", RED,
        RGBColor(0xFD, 0xEC, 0xEE), 12.5, 11.5)
callout(s, Inches(0.55), Inches(5.05), Inches(7.5), Inches(1.5),
        "ALSO NOTED",
        "The attached “HTML” file is a 553-byte macOS AppleDouble metadata stub, not notebook HTML — the "
        "executable-notebook requirement was therefore unmet in the original zip. A freshly executed "
        "notebook and a genuine HTML export are now supplied. The report also covers all 4 challenges "
        "although a 3-member team selects 3.", ORANGE, LIGHT, 12.5, 11.5)

# ---------------------------------------------------------------- SLIDE 4
s = new(4, "Data pipeline — why the original code stopped running",
        "Two silent yfinance changes plus a scale mismatch corrupted every downstream table")
rows = [["Defect in fetch_data()", "Symptom visible in the submitted PDF", "Fix"],
      ["MultiIndex ('Price','Ticker') columns", "Regressor named “BTC-USD”, not Volume; ECM rows print as ('lag_eg_resid','')",
       "Flatten with get_level_values(0)"],
      ["auto_adjust default flipped to True", "Every close silently changed → results not reproducible",
       "Pin auto_adjust=False"],
      ["No index frequency declared", "Repeated ValueWarning: No frequency information",
       "asfreq('D') for 24/7 crypto"],
      ["Volume ~1e10 beside price changes ~1e2", "Condition number 7.7e10; VAR coefficients print 0.000000",
       "Volume in billions, returns in percent"],
      ["Live download only, no cache", "App dies when Yahoo rate-limits a cloud IP",
       "Committed CSV + graceful fallback"]]
table(s, Inches(0.55), Inches(1.4), Inches(12.2), rows, col_w=[3.0, 5.2, 3.0],
      size=11, row_h=0.62, head_h=0.4)
callout(s, Inches(0.55), Inches(5.15), Inches(5.95), Inches(1.55),
        "REPRODUCIBILITY CONTRACT",
        "One prepare() function is the single source of truth for every model: flat columns, daily "
        "frequency, percent returns, volume in billions, no in-place mutation of a shared frame, "
        "seed fixed at 42.", GREEN, RGBColor(0xE8, 0xF6, 0xED), 13, 12)
callout(s, Inches(6.8), Inches(5.15), Inches(5.95), Inches(1.55),
        "RESULT",
        f"{R['n_obs']:,} daily observations, {R['window'][0]} → {R['window'][1]}. "
        "The whole pipeline re-runs headlessly with one command and writes 10 figures plus a "
        "machine-readable results.json.", BLUE, LIGHT, 13, 12)

# ---------------------------------------------------------------- SLIDE 5
s = new(5, "Challenge 1a — Establishing the integration order",
        "ADF (H₀: unit root) and KPSS (H₀: stationary) must be read jointly")
tb(s, Inches(0.55), Inches(1.3), Inches(6.0), Inches(0.3),
   "Δyₜ = α + βt + φyₜ₋₁ + Σᵢ γᵢ Δyₜ₋ᵢ + εₜ", 15, NAVY2, True)
rows = [["Series", "ADF p", "KPSS p", "Verdict"]]
label = {"Close": "Close (level)", "lnClose": "ln Close", "Volume": "Volume (level)",
         "dClose": "Δ Close", "ret_pct": "Return (%)", "dVol_bn": "Δ Volume (bn)"}
for u in R["unit_roots"]:
    if u["series"] in label:
        v = u["verdict"].replace("conflicting - possible fractionally integrated / structural break",
                                 "conflicting")
        rows.append([label[u["series"]], f"{u['ADF_p']:.4g}", u["KPSS_p"], v])
table(s, Inches(0.55), Inches(1.78), Inches(6.35), rows, col_w=[1.6, 1.0, 0.95, 2.5],
      size=10.5, row_h=0.36, head_h=0.34)
pic(s, FIG / "01_stationarity.png", Inches(7.15), Inches(1.35), w=Inches(5.6))
callout(s, Inches(0.55), Inches(4.42), Inches(6.35), Inches(1.05),
        "FIX · KPSS p-values are bounded",
        "statsmodels truncates them to [0.01, 0.10]. The original printed “p: 0.1” as an exact value; "
        "we report the bound and classify on the joint outcome.", ORANGE, LIGHT, 12.5, 11.5)
callout(s, Inches(0.55), Inches(5.62), Inches(12.2), Inches(1.15),
        "DECISIVE FINDING · the partner series was never tested",
        "Close is I(1) but Volume is already stationary in levels (ADF p = 0.004). Cointegration is only "
        "defined between series of the same order — so the BTC/Volume regression is unbalanced, and the "
        "correct verdict is “precondition violated”, not merely “no cointegration”.", RED,
        RGBColor(0xFD, 0xEC, 0xEE), 13, 12)

# ---------------------------------------------------------------- SLIDE 6
s = new(6, "Challenge 1b — A textbook spurious regression",
        "Close ~ Volume looks like a finding, and is not one")
eg = R["eg_btc_volume"]
kpi(s, Inches(0.55), Inches(1.32), Inches(2.15), Inches(1.1), f"{eg['r2']:.3f}", "R² — looks\nmeaningful", NAVY)
kpi(s, Inches(2.84), Inches(1.32), Inches(2.15), Inches(1.1), f"{eg['dw']:.3f}", "Durbin–Watson\nnear zero", RED)
kpi(s, Inches(5.13), Inches(1.32), Inches(2.15), Inches(1.1), f"{eg['eg_p']:.3f}", "Engle–Granger p\nresidual unit root", RED)
pic(s, FIG / "02_spurious.png", Inches(7.5), Inches(1.3), w=Inches(5.25))
callout(s, Inches(0.55), Inches(2.62), Inches(6.73), Inches(1.5),
        "FIX · wrong critical values were used",
        f"The original ran a plain ADF on the residuals and reported p = {eg['naive_adf_p']:.4f}. "
        f"Residual-based tests need MacKinnon cointegration critical values: the correct p is "
        f"{eg['eg_p']:.4f}. statsmodels' coint() was imported but never called.", ORANGE, LIGHT, 13, 12)
callout(s, Inches(0.55), Inches(4.28), Inches(6.73), Inches(1.2),
        "FIX · an ECM was fitted without cointegration",
        f"φ = {R['ecm_btc_volume']['phi']:.5f} (p = {R['ecm_btc_volume']['phi_p']:.3f}) — no error "
        "correction. The Granger representation theorem says an ECM requires cointegration; fitting one "
        "anyway is invalid.", RED, RGBColor(0xFD, 0xEC, 0xEE), 13, 12)
callout(s, Inches(0.55), Inches(5.64), Inches(12.2), Inches(1.12),
        "ALSO CORRECTED · a report mis-attribution",
        "The report blames “an automated lag order of 15” for the ECM's failure. The ECM has only two "
        "regressors — the 15 lags belong to the VAR. And a condition number of 7.7e10 with a single "
        "regressor is a units problem, not multicollinearity.", ORANGE, LIGHT, 13, 12)

# ---------------------------------------------------------------- SLIDE 7
s = new(7, "Challenge 1c — A genuine equilibrium: Alphabet's dual-class shares",
        "GOOGL and GOOG claim the same cash flows, so theory predicts a log hedge ratio of ≈ 1")
e2, m2 = R["eg_googl_goog"], R["ecm_googl_goog"]
kpi(s, Inches(0.55), Inches(1.32), Inches(2.0), Inches(1.1), f"{e2['beta']:.4f}", "hedge ratio β\ntheory ≈ 1", GREEN)
kpi(s, Inches(2.69), Inches(1.32), Inches(2.0), Inches(1.1), f"{e2['eg_p']:.4f}", "EG p-value\ncointegrated", GREEN)
kpi(s, Inches(4.83), Inches(1.32), Inches(2.0), Inches(1.1), f"{m2['phi']:.4f}", "adjustment φ\np = 2.5e-09", GREEN)
kpi(s, Inches(6.97), Inches(1.32), Inches(2.0), Inches(1.1), f"{m2['half_life_days']:.1f} d", "half-life of\nthe spread", GREEN)
pic(s, FIG / "03_ecm.png", Inches(9.2), Inches(1.32), w=Inches(3.55))
tb(s, Inches(0.55), Inches(2.66), Inches(8.4), Inches(0.35),
   "Δyₜ = α + φ (yₜ₋₁ − β̂ xₜ₋₁) + γ Δxₜ + εₜ ,   φ < 0", 15, NAVY2, True)
bullets(s, Inches(0.55), Inches(3.15), Inches(8.4), Inches(2.1), [
    ("Both series are I(1) — ", "the precondition the BTC/Volume test violated is satisfied here."),
    (f"{abs(m2['phi'])*100:.1f}% of any gap closes each day. ", f"That is a half-life of {m2['half_life_days']:.1f} days, estimated with p = 2.5×10⁻⁹."),
    ("Short-run γ ≈ 1.00 — ", "the two share classes move essentially one-for-one intraday."),
    ("This is the positive example the brief asks for — ", "the original submission only ever reported a rejection."),
], size=13.5, gap=11)
callout(s, Inches(0.55), Inches(5.5), Inches(12.2), Inches(1.26),
        "DEPLOYMENT",
        "A tradable statistical-arbitrage signal: enter when the log spread breaches a threshold, size for a "
        "~12-day reversion, and re-test cointegration monthly because dual-class spreads can structurally "
        "break. This is how the equilibrium model becomes a decision rather than a table.", GREEN,
        RGBColor(0xE8, 0xF6, 0xED), 13, 12)

# ---------------------------------------------------------------- SLIDE 8
s = new(8, "Challenge 1d — VAR: parsimony and the diagnostics that were missing",
        "Stationary, scale-matched inputs; data-driven lag cap; stability and whiteness tests")
v = R["var"]
kpi(s, Inches(0.55), Inches(1.32), Inches(2.1), Inches(1.1), f"{v['maxlags']}", "data-driven lag cap\n(Schwert rule)", NAVY)
kpi(s, Inches(2.79), Inches(1.32), Inches(2.1), Inches(1.1), f"{v['p']}", f"lags chosen by BIC\nwas {15} (AIC at cap)", GREEN)
kpi(s, Inches(5.03), Inches(1.32), Inches(2.1), Inches(1.1), "Yes", "all roots inside\nthe unit circle", GREEN)
kpi(s, Inches(7.27), Inches(1.32), Inches(2.1), Inches(1.1), "1.1e-09", "whiteness p\nresiduals NOT white", RED)
pic(s, FIG / "04_var_irf.png", Inches(9.6), Inches(1.32), w=Inches(3.15))
callout(s, Inches(0.55), Inches(2.62), Inches(8.8), Inches(1.32),
        "FIX · the lag cap was arbitrary and binding",
        "max_lags = min(15, n−1) made AIC, HQIC and FPE all select exactly 15 — the ceiling, not an interior "
        f"optimum. With the Schwert cap of {v['maxlags']}, AIC still wants {v['orders']['aic']} while BIC "
        f"settles on {v['p']}: evidence of AIC over-fitting, exactly as the report suspected.", ORANGE, LIGHT, 13, 12)
callout(s, Inches(0.55), Inches(4.1), Inches(8.8), Inches(1.32),
        "FIX · no residual diagnostic was ever run",
        f"The Portmanteau test rejects white residuals (p = {v['whiteness_p']:.2g}), so linear dynamics remain "
        f"unmodelled — a materially worse problem than over-fitting. Granger causality does run from volume "
        f"to price (p = {v['granger']['p']:.4f}), which is the economically interesting result.", RED,
        RGBColor(0xFD, 0xEC, 0xEE), 13, 12)
callout(s, Inches(0.55), Inches(5.58), Inches(12.2), Inches(1.18),
        "SCALE FIX",
        "Expressing volume in billions instead of raw units cut the condition number from 7.7×10¹⁰ to a "
        "readable magnitude, so impulse responses and coefficients are interpretable rather than printing "
        "as 0.000000.", BLUE, LIGHT, 13, 12)

# ---------------------------------------------------------------- SLIDE 9
s = new(9, "Challenge 2a — The single most serious defect in the submission",
        "result.predict() is a conditional mean in dollars — not a regime label, not a probability")
rect(s, Inches(0.55), Inches(1.3), Inches(6.1), Inches(1.5), RGBColor(0xFD, 0xEC, 0xEE))
tb(s, Inches(0.75), Inches(1.44), Inches(5.7), Inches(0.3), "WHAT THE CODE DID", 12.5, RED, True)
tb(s, Inches(0.75), Inches(1.78), Inches(5.7), Inches(0.4),
   "df['Regime'] = regime_result.predict()", 14, NAVY, True, font="Consolas")
tb(s, Inches(0.75), Inches(2.18), Inches(5.7), Inches(0.55),
   "The report then wrote: “a value of 60 implies a 60% probability of being in Regime 1.”", 12, NAVY)
rect(s, Inches(6.9), Inches(1.3), Inches(5.85), Inches(1.5), RGBColor(0xE8, 0xF6, 0xED))
tb(s, Inches(7.1), Inches(1.44), Inches(5.45), Inches(0.3), "WHAT IT MUST DO", 12.5, GREEN, True)
tb(s, Inches(7.1), Inches(1.78), Inches(5.45), Inches(0.4),
   "result.smoothed_marginal_probabilities[1]", 13, NAVY, True, font="Consolas")
tb(s, Inches(7.1), Inches(2.18), Inches(5.45), Inches(0.55),
   "Bounded on [0,1]; classify with argmax across regimes.", 12, NAVY)
md_ = R["markov_dollar"]
callout(s, Inches(0.55), Inches(2.98), Inches(12.2), Inches(1.0),
        "PROOF",
        f"On the submitted specification predict() spans exactly {md_['predict_range'][0]:.2f} to "
        f"{md_['predict_range'][1]:.2f} dollars — the two fitted regime means (21.06 and 58.79). "
        "That is precisely the “20–60” range the report mistook for percentages. It can never be a probability.",
        RED, RGBColor(0xFD, 0xEC, 0xEE), 13, 12)
mr = R["markov_returns"]["regimes"]
rows = [["Corrected model on percent returns", "Mean (%/day)", "Std dev (%/day)",
         "Expected duration", "Share of days"],
        ["Regime 0 — calm", f"{mr[0]['mu']:.3f}", f"{mr[0]['sigma']:.2f}",
         f"{mr[0]['expected_duration']:.1f} days", f"{mr[0]['share']:.1%}"],
        ["Regime 1 — turbulent", f"{mr[1]['mu']:.3f}", f"{mr[1]['sigma']:.2f}",
         f"{mr[1]['expected_duration']:.1f} days", f"{mr[1]['share']:.1%}"]]
table(s, Inches(0.55), Inches(4.18), Inches(12.2), rows, col_w=[3.4, 1.9, 2.1, 2.4, 2.0],
      size=12, row_h=0.44, head_h=0.5)
bullets(s, Inches(0.55), Inches(5.65), Inches(12.2), Inches(1.2), [
    ("Two further report contradictions resolved: ",
     "the turbulent regime has a HIGHER mean (+0.38% vs +0.08%), not “significant negative price changes”; "
     "and both variances are significant at p < 0.001, so “variance differences are not significant” is wrong."),
    ("Also fixed: ", "a 12-start likelihood search (the Markov likelihood is multimodal) and regime "
     "re-ordering so “0 = calm, 1 = turbulent” always holds."),
], size=12.5, gap=7)

# ---------------------------------------------------------------- SLIDE 10
s = new(10, "Challenge 2b — Dollar differences confuse the price level with risk",
        "BTC rose from ~$7k to ~$90k: the same percentage move became a far bigger dollar move")
pic(s, FIG / "05b_regime_artefact.png", Inches(0.55), Inches(1.4), w=Inches(6.5))
by = R["highvol_share_by_year"]
rows = [["Year", "Dollar spec", "Return spec"],
        ["2020", f"{by['dollar_spec']['2020']:.0%}", f"{by['returns_spec']['2020']:.0%}"],
        ["2021", f"{by['dollar_spec']['2021']:.0%}", f"{by['returns_spec']['2021']:.0%}"],
        ["2023", f"{by['dollar_spec']['2023']:.0%}", f"{by['returns_spec']['2023']:.0%}"],
        ["2024", f"{by['dollar_spec']['2024']:.0%}", f"{by['returns_spec']['2024']:.0%}"]]
table(s, Inches(7.35), Inches(1.5), Inches(5.4), rows, col_w=[1.2, 2.1, 2.1],
      size=12, row_h=0.4, head_h=0.38, highlight={1, 4})
callout(s, Inches(7.35), Inches(3.75), Inches(5.4), Inches(1.35),
        "ECONOMICALLY BACKWARDS",
        "The dollar model calls 2021 turbulent on 100% of days and 2024–25 on ~95%, while labelling the "
        "March-2020 crash calm on 93% of days. It is tracking the price level, not volatility.", RED,
        RGBColor(0xFD, 0xEC, 0xEE), 13, 12)
callout(s, Inches(7.35), Inches(5.25), Inches(5.4), Inches(1.5),
        "CORRECTED READING",
        "On percent returns the turbulent share is 25% in 2020, 44% in 2021 and 13% in 2023. Caveat: the states "
        "last only ~4 and ~2 days, so this is a fat-tail mixture, not persistent regimes — use GARCH for risk.", GREEN,
        RGBColor(0xE8, 0xF6, 0xED), 13, 12)
callout(s, Inches(0.55), Inches(5.55), Inches(6.5), Inches(1.2),
        "WHY IT MATTERS (ILLUSTRATION)",
        "Any rule driven by the dollar-based labels would have treated the March-2020 crash as calm and 2024–25 "
        "as permanently turbulent — the opposite of what realised volatility showed.", ORANGE,
        LIGHT, 13, 12)

# ---------------------------------------------------------------- SLIDE 11
s = new(11, "Challenge 3a — From a vacuous test to a real diagnosis",
        "The original VIF could not have detected collinearity even if it were present")
callout(s, Inches(0.55), Inches(1.3), Inches(6.1), Inches(1.28),
        "DEFECT 1 · VIF on the wrong variables",
        "VIF was computed on Diff_Close and Diff_Volume — the two DEPENDENT variables of the VAR, which are "
        "not regressors of one another. VIF ≈ 1.001 was guaranteed. A VIF was also reported for the intercept.",
        RED, RGBColor(0xFD, 0xEC, 0xEE), 13, 11.5)
callout(s, Inches(0.55), Inches(2.72), Inches(6.1), Inches(1.28),
        "DEFECT 2 · PCA without standardisation",
        "Volume (~1e10) dwarfs price changes (~1e2), so “PC1 = 100%” was a units artefact. Standardised, the "
        "split is 51.9% / 48.1% and the correlation is −0.04: there is no common factor at all.",
        RED, RGBColor(0xFD, 0xEC, 0xEE), 13, 11.5)
callout(s, Inches(0.55), Inches(4.14), Inches(6.1), Inches(1.28),
        "DEFECT 3 · look-ahead bias",
        "A gap built as rolling(w).mean()/Close − 1 contains today's return. Regressing today's return on it "
        "is circular: R² ≈ 0.57 collapses to ≈ 0.008 once every predictor is lagged one day.",
        RED, RGBColor(0xFD, 0xEC, 0xEE), 13, 11.5)
pic(s, FIG / "06_collinearity.png", Inches(6.9), Inches(1.35), w=Inches(5.85))
rem = R["remedies"]
rows = [["Real predictor set (all lagged 1 day)", "Value"],
        ["Worst VIF — ma10_gap", f"{R['vif'][0]['VIF']:.1f}"],
        ["ma20_gap / ma50_gap", f"{R['vif'][1]['VIF']:.1f} / {R['vif'][2]['VIF']:.1f}"],
        ["Significant coefficients at 5%", f"{rem['n_sig']} of {rem['n_pred']}"],
        ["In-sample R²", f"{rem['ols_r2']:.4f}"],
        ["Condition number, raw → scaled", f"{R['pca']['cond_raw']:.0f} → {R['pca']['cond_scaled']:.1f}"]]
table(s, Inches(6.9), Inches(4.05), Inches(5.85), rows, col_w=[3.9, 1.9],
      size=11.5, row_h=0.36, head_h=0.36)
tb(s, Inches(0.55), Inches(5.6), Inches(12.2), Inches(1.15),
   "The report asserted both “one dominant factor showing alignment” and “no redundant information”. "
   "Those claims contradict each other, and neither follows from the test that was run. "
   "With overlapping moving averages the pathology is unmistakable — and now measurable.", 13, NAVY)

# ---------------------------------------------------------------- SLIDE 12
s = new(12, "Challenge 3b — Real collinearity, tiny forecast gain",
        "Expanding-window time-series cross-validation — a random K-fold would leak the future")
pic(s, FIG / "07_remedies.png", Inches(0.55), Inches(1.4), w=Inches(6.2))
rows = [["Model", "OOS MSE", "Gain vs OLS"]]
for k in ["OLS", "Trimmed OLS", "Ridge", "PCR3"]:
    rows.append([k, f"{rem['mse'][k]:.4f}", f"{rem['gain_pct'][k]:+.2f}%"])
table(s, Inches(7.05), Inches(1.5), Inches(5.7), rows, col_w=[2.4, 1.7, 1.6],
      size=12, row_h=0.42, head_h=0.4, highlight={3, 4})
bullets(s, Inches(7.05), Inches(3.75), Inches(5.7), Inches(1.7), [
    ("Trim ", "ma10_gap and ma20_gap until every VIF ≤ 10."),
    ("Ridge ", f"penalty selected by CV (α = {rem['ridge_alpha']:.0f}) shrinks the redundant directions."),
    ("PCR ", f"3 components carry ~85% of predictor variance; {R['pca']['n_for_90']} reach 90%."),
], size=13, gap=10)
callout(s, Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.4),
        "INTERPRETATION — AND AN HONEST CAVEAT",
        "Principal-component regression wins, Ridge is second, and the kitchen-sink OLS is worst: the ordering "
        "is exactly what collinearity theory predicts. The absolute gains are small (0.5–0.7%) because daily "
        "BTC returns are close to unforecastable — the point is the ranking and the stability of the "
        "coefficients, not a promise of alpha.", BLUE, LIGHT, 13, 12)

# ---------------------------------------------------------------- SLIDE 13
s = new(13, "Challenge 4 — Feature extraction, and a conclusion that reverses",
        "ARMA conditional mean · GARCH conditional variance · squared returns · Kalman local level")
pic(s, FIG / "08_features.png", Inches(0.55), Inches(1.3), h=Inches(5.45))
f = R["features"]; gp = f["garch_params"]
rows = [["GARCH(1,1)-t on percent returns", "Estimate"],
        ["ω (omega)", f"{gp['omega']:.4f}"],
        ["α (ARCH)", f"{gp['alpha[1]']:.4f}"],
        ["β (GARCH)", f"{gp['beta[1]']:.4f}"],
        ["Persistence α+β", f"{f['persistence']:.4f}"],
        ["Tail index ν", f"{gp['nu']:.2f}"],
        ["Volatility half-life", "infinite"]]
table(s, Inches(5.3), Inches(1.35), Inches(4.0), rows, col_w=[2.6, 1.4],
      size=11, row_h=0.34, head_h=0.42, highlight={4})
callout(s, Inches(9.5), Inches(1.35), Inches(3.25), Inches(2.6),
        "THE REVERSAL",
        "The original reported α = 0.673, β = 0.036 (sum 0.71) and concluded “baseline level volatility … "
        "no clustering”.\n\narch had warned: DataScaleWarning, scale of y is 0.0011. On percent returns every "
        "t-error specification is very persistent (0.99–1.00); GARCH-t sits on the optimiser's α+β ≤ 1 bound, "
        "EGARCH-t (unconstrained, best AIC) gives 0.99.",
        RED, RGBColor(0xFD, 0xEC, 0xEE), 12.5, 11)
rows2 = [["Specification (same data)", "AIC", "α+β"]]
for sp in f["spec_table"]:
    rows2.append([sp["spec"], f"{sp['AIC']:.1f}", f"{sp['persistence']:.4f}"])
table(s, Inches(5.3), Inches(4.15), Inches(7.45), rows2, col_w=[3.6, 1.9, 1.9],
      size=11, row_h=0.36, head_h=0.36, highlight={1})
tb(s, Inches(5.3), Inches(6.05), Inches(7.45), Inches(0.85),
   f"EGARCH-t wins on AIC. Kalman signal-to-noise = {f['kalman']['signal_to_noise']:.1e} → no persistent "
   f"drift. ν ≈ 3: Gaussian VaR understates the 99% tail (not the 95%).", 12, NAVY)

# ---------------------------------------------------------------- SLIDE 14
s = new(14, "From model to decision — the deployable risk output",
        "Student-t Value-at-Risk, not Gaussian, because ν ≈ 3")
pic(s, FIG / "09_risk.png", Inches(0.55), Inches(1.35), w=Inches(6.3))
rf = R["risk_forecast"][0]
kpi(s, Inches(7.15), Inches(1.4), Inches(2.7), Inches(1.15), f"{rf['vol_forecast_pct']:.2f}%",
    "1-day volatility\nforecast", NAVY)
kpi(s, Inches(10.0), Inches(1.4), Inches(2.75), Inches(1.15), f"{-rf['VaR95_pct']:.1f}%",
    "1-day 95% VaR\n(loss)", ORANGE)
kpi(s, Inches(7.15), Inches(2.7), Inches(2.7), Inches(1.15), f"{-rf['VaR99_pct']:.1f}%",
    "1-day 99% VaR\n(loss)", RED)
kpi(s, Inches(10.0), Inches(2.7), Inches(2.75), Inches(1.15), "10 d", "term structure\npublished daily", BLUE)
bullets(s, Inches(7.15), Inches(4.05), Inches(5.6), Inches(1.6), [
    ("Monitor ", "1-day 99% VaR breaches as a model-health check."),
    ("Read ", "P(turbulent) as a descriptive flag, not a trading signal."),
    ("Note ", "educational illustration only — not investment advice."),
], size=13, gap=9)
callout(s, Inches(0.55), Inches(5.2), Inches(12.2), Inches(1.55),
        "WHAT WAS DELIVERED ALONGSIDE THIS DECK",
        "A corrected library (5 modules), an executed notebook with zero errors plus a real HTML export, "
        "a headless reproduction script writing 10 figures and results.json, a 9-page app with live "
        "CoinGecko data, a simulation playground and a side-by-side Bug lab, cached data for offline use, and a full defect register in "
        "report/REVIEW.md.", GREEN, RGBColor(0xE8, 0xF6, 0xED), 13, 12)

# ---------------------------------------------------------------- SLIDE 15
s = new(15, "Deploy it free — so any visitor can run the models",
        "Same repository, five free routes; no API key, no credit card")
rows = [["Platform", "What the visitor gets", "Free tier", "Effort"],
        ["Streamlit Community Cloud", "The full interactive app, public URL", "1 public app, unlimited viewers", "≈ 5 min"],
        ["Hugging Face Spaces", "Same app on HF infrastructure", "2 vCPU / 16 GB, always on", "≈ 10 min"],
        ["Google Colab", "Editable notebook, run all cells", "Free CPU, GPU optional", "1 click"],
        ["Binder (mybinder.org)", "Live JupyterLab from the repo", "Free, session-limited", "1 click"],
        ["GitHub Codespaces", "Full VS Code in the browser", "60 h/month personal", "1 click"]]
table(s, Inches(0.55), Inches(1.35), Inches(12.2), rows, col_w=[3.0, 4.3, 3.2, 1.3],
      size=11.5, row_h=0.44, head_h=0.38, highlight={1})
tb(s, Inches(0.55), Inches(4.05), Inches(6.0), Inches(0.32), "Fastest path — Streamlit Cloud", 15, NAVY, True)
bullets(s, Inches(0.55), Inches(4.45), Inches(6.0), Inches(2.2), [
    "Push the folder to a public GitHub repo.",
    "Go to share.streamlit.io and sign in with GitHub.",
    "New app → pick repo, branch main, file app.py.",
    "Deploy. requirements.txt is read automatically.",
    "Share the URL — visitors change inputs and re-run live.",
], size=12.5, gap=7)
callout(s, Inches(6.85), Inches(4.05), Inches(5.9), Inches(1.35),
        "THE ONE THING THAT BREAKS FREE DEPLOYMENTS",
        "Yahoo Finance routinely blocks cloud IPs. The committed CSV cache means the app still runs; live "
        "download is opt-in and falls back gracefully instead of crashing.", ORANGE, LIGHT, 13, 12)
callout(s, Inches(6.85), Inches(5.55), Inches(5.9), Inches(1.2),
        "NEXT STEPS",
        "Backtest VaR (Kupiec / Christoffersen); add a FIGARCH component model; extend the cointegration "
        "screen to more pairs; trim the written report to the 5–6 page limit.", BLUE, LIGHT, 13, 12)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))
