# 📈 BTC Econometrics Lab

> *Where real-time market data meets classical econometrics.*

A multi-page Streamlit app built from the **MScFE 610 Financial Econometrics — Group Work Project 2 (Group 8665)** coursework.
It re-runs four econometric challenges on daily BTC-USD data (non-stationarity & equilibrium, regime switching, multicollinearity,
feature extraction) and applies the fitted **GARCH(1,1)-t** and **2-state Markov-switching** models to **live** CoinGecko prices
to show next-day volatility, a turbulence score and a one-day VaR.

🔗 **Live app:** `https://<your-app-name>.streamlit.app` *(add the URL after you deploy)*

## ⚠️ Disclaimer
**Educational and research use only.** Nothing here is financial, investment, trading, legal, tax, medical or other professional advice, or a
recommendation to buy, sell, hold or trade any asset. Live data are third-party and may be wrong or delayed; model outputs can be wrong or misleading and
do **not** predict price direction. The developer and co-authors accept no liability for decisions made on the basis of this app.

## ✨ Features
| Page | What it does |
|---|---|
| 🏠 Home | Live BTC/ETH prices, 5 live model stats, live price + P(turbulent) chart, 4 insight cards, VaR, auto-refresh every 60 s |
| 🧠 About | What the model does (and does not), honest accuracy notes, developer bio |
| 🎓 Learn | How to use the simulation, score interpretation table, FAQ, glossary |
| 🎛 Simulate | Live playground: shock, σ, α/β/ν stress, horizon, VaR level → metrics, contribution chart, volatility path, history, CSV download |
| 📊 Insights | Daily tip + commentary from current live data |
| 📈 Econometrics Lab | The full workbench: ADF/KPSS/EG/ECM/VAR · Markov switching · VIF/PCA/Ridge · ARMA/GARCH/Kalman · Bug lab · Method notes |
| 📁 Docs | Report.pdf, Presentation.pptx, REVIEW.md, results.json, notebook HTML; model card; API table; version history |
| 🧭 Journey | Project timeline and figures |
| 📞 Contact | Developer card, links, feedback form (mailto) |

Resilience: every live call is wrapped in `try/except` and falls back to the committed CSV cache (clearly labelled "OFFLINE FALLBACK");
`model/model.pkl` falls back to a built-in copy of the fitted parameters.

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy free on Streamlit Community Cloud
1. Push this folder to a **public** GitHub repo (e.g. `btc-econometrics-lab`).
2. Go to <https://share.streamlit.io> → sign in with GitHub → **New app**.
3. Pick the repo, branch `main`, main file `app.py` → **Deploy**.
4. Every `git push` to `main` redeploys automatically. No API key or secret is required.

## 🔁 Rebuild the artefacts (optional)
```bash
pip install -r requirements-dev.txt
python scripts/run_all.py          # figures + report/results.json
python scripts/train_model.py      # model/model.pkl (validated against arch/statsmodels)
python scripts/build_notebook.py   # executed notebook (then: jupyter nbconvert --to html ...)
python scripts/build_slides.py     # assets/docs/Presentation.pptx
python scripts/build_report.py     # assets/docs/Report.pdf
```

## 🗂 Structure
```
app.py                  Home          pages/   8 pages (see table)
src/                    estimation library (stationarity, regime, collinearity, features, data)
utils/                  layout, styles, disclaimer, live data, engine (filters), insights, model store
model/                  model.pkl (fitted parameters) + preprocess.py
data/                   cached Yahoo Finance CSVs      report/  results.json, REVIEW.md, tables
notebooks/              executed notebook + HTML       figures/ 10 PNGs      assets/docs/ Report.pdf, Presentation.pptx
scripts/                run_all, train_model, build_*  .streamlit/config.toml  (dark theme)
```

## 👤 Author
**Ajeet Kumar** — [LinkedIn](https://www.linkedin.com/in/ajeetk095) · [GitHub](https://github.com/ajeetk095) · [Email](mailto:ajeetk095@gmail.com)
Original coursework with Kow Kai Sheng and Faith Nyaboe Momanyi.

## 📄 License
MIT — see `LICENSE`.
