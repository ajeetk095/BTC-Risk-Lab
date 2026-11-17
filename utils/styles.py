"""Dark glassmorphism theme, injected once per page."""
import streamlit as st

CSS = """
<style>
  :root { --g1:#6366f1; --g2:#8b5cf6; --g3:#06b6d4; --btc:#f7931a; }
  .stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.18), transparent 60%),
                       radial-gradient(900px 500px at 100% 0%, rgba(6,182,212,.12), transparent 55%), #0f172a; }
  .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
  h1, h2, h3 { letter-spacing: -.01em; }
  footer, #MainMenu { visibility: hidden; }

  /* metric cards */
  div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: .85rem 1rem;
    backdrop-filter: blur(8px); transition: transform .18s ease, box-shadow .18s ease;
  }
  div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(99,102,241,.25); }
  div[data-testid="stMetricValue"] { font-size: 1.45rem; }

  /* buttons */
  .stButton > button, .stDownloadButton > button, a[data-testid="stLinkButton"], div[data-testid="stLinkButton"] a {
    background: linear-gradient(90deg, var(--g1), var(--g2) 55%, var(--g3)); color: #fff !important;
    border: 0; border-radius: 12px; font-weight: 600; transition: box-shadow .18s ease, transform .18s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stLinkButton"] a:hover {
    box-shadow: 0 0 22px rgba(139,92,246,.55); transform: translateY(-1px); color: #fff !important; }

  /* hero + cards */
  .hero { background: linear-gradient(120deg, rgba(99,102,241,.35), rgba(139,92,246,.28) 50%, rgba(6,182,212,.28));
          border: 1px solid rgba(255,255,255,.14); border-radius: 20px; padding: 1.6rem 1.8rem; margin-bottom: .8rem; }
  .hero h1 { margin: 0 0 .2rem 0; font-size: 2.1rem; font-weight: 800; }
  .hero p  { margin: .1rem 0; opacity: .92; }
  .hero .warn { color: #fca5a5; font-weight: 600; font-size: .92rem; margin-top: .6rem; }
  .glass { background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
           border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .8rem; }
  .glass:hover { transform: translateY(-2px); transition: transform .18s ease; }
  .insight-success { border-left: 5px solid #22c55e; } .insight-info { border-left: 5px solid #38bdf8; }
  .insight-warning { border-left: 5px solid #f59e0b; }
  .insight-title { font-weight: 700; margin-bottom: .2rem; }

  /* sidebar */
  section[data-testid="stSidebar"] { background: rgba(15,23,42,.92); border-right: 1px solid rgba(255,255,255,.08); }
  .side-disclaimer { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.45); border-radius: 12px;
                     padding: .6rem .75rem; font-size: .8rem; color: #fecaca; margin-bottom: .6rem; }

  /* verdict boxes (dark-theme friendly) */
  .verdict-ok   { background: rgba(34,197,94,.13);  border-left: 5px solid #22c55e; padding:.7rem 1rem; border-radius: 8px; color:#bbf7d0; }
  .verdict-bad  { background: rgba(239,68,68,.13);  border-left: 5px solid #ef4444; padding:.7rem 1rem; border-radius: 8px; color:#fecaca; }
  .verdict-warn { background: rgba(245,158,11,.13); border-left: 5px solid #f59e0b; padding:.7rem 1rem; border-radius: 8px; color:#fde68a; }
  .verdict-ok code, .verdict-bad code, .verdict-warn code { color: inherit; background: rgba(255,255,255,.1); }
  .small { font-size: .85rem; opacity: .75; }

  /* timeline */
  .tl { position: relative; margin: 1rem 0 1rem .6rem; padding-left: 1.4rem; border-left: 2px solid rgba(139,92,246,.6); }
  .tl-item { margin-bottom: 1.1rem; position: relative; }
  .tl-item::before { content: ""; position: absolute; left: -1.95rem; top: .35rem; width: 14px; height: 14px; border-radius: 50%;
                     background: linear-gradient(135deg, var(--g1), var(--g3)); box-shadow: 0 0 10px rgba(99,102,241,.7); }
  .tl-date { font-size: .8rem; opacity: .7; } .tl-title { font-weight: 700; }

  .footer-disclaimer { text-align: center; font-size: .78rem; color: #94a3b8; margin-top: .6rem; }
  @media (max-width: 640px) { .hero h1 { font-size: 1.5rem; } .block-container { padding-left: .8rem; padding-right: .8rem; } }
  a:focus-visible, button:focus-visible { outline: 3px solid #38bdf8 !important; outline-offset: 2px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
