"""Rule-based commentary on the model state. Educational wording only - never advice."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List


def generate_advice(a: dict, change_24h: float | None = None, live: bool = True) -> List[Dict[str, str]]:
    """Four insight cards from an `engine.analyse` result. type: success | info | warning."""
    out: List[Dict[str, str]] = []
    s, conf = a["score"], a["confidence"]

    if s >= 65:
        out.append(dict(type="warning", title=f"Turbulent state likely ({s:.0f}/100)",
                        body="The filtered probability of the high-variance state is high. Historically, days in this "
                             "state had a standard deviation of roughly 5% versus about 1.6% in the calm state."))
    elif s <= 35:
        out.append(dict(type="success", title=f"Calm state likely ({s:.0f}/100)",
                        body="The model assigns most weight to the low-variance state. Calm states in this sample lasted "
                             "only about four days on average, so this label can flip quickly."))
    else:
        out.append(dict(type="info", title=f"Mixed signal ({s:.0f}/100)",
                        body="Neither state clearly dominates. When the score sits mid-range the regime label carries "
                             "little information — lean on the volatility forecast instead."))

    p = a["vol_pctile"]
    kind = "warning" if p >= 75 else ("success" if p <= 25 else "info")
    out.append(dict(type=kind, title=f"Forecast volatility {a['sigma_next']:.2f}% / day",
                    body=f"That is higher than {p:.0f}% of days in the 2020–2025 training sample "
                         f"(30-day realised: {a['realised_30d']:.2f}%). GARCH volatility is persistent, so a high reading "
                         "tends to decay slowly rather than vanish overnight."))

    out.append(dict(type="info", title=f"Model 1-day 99% VaR ≈ {a['var99']:.1f}%",
                    body=f"With fat tails (Student-t) the 99% one-day loss threshold is {a['var99']:.1f}% versus "
                         f"{a['var99_normal']:.1f}% under a normal assumption; at 95% they are {a['var95']:.1f}% and "
                         f"{a['var95_normal']:.1f}%. VaR is a statistical threshold, not a worst case."))

    if not live:
        out.append(dict(type="warning", title="Running on offline fallback data",
                        body="The live API could not be reached, so every number reflects cached history, not today's market."))
    elif conf < 60:
        out.append(dict(type="warning", title=f"Low state confidence ({conf:.0f}%)",
                        body="The regime probability is close to 50/50 — treat today's regime label as unreliable."))
    else:
        chg = f" (24h move {change_24h:+.1f}%)" if change_24h is not None else ""
        out.append(dict(type="success", title=f"State confidence {conf:.0f}%",
                        body=f"The model is fairly decisive about today's state{chg}. Confidence describes the state "
                             "classification only — it says nothing about future price direction."))
    return out[:4]


TIPS = [
    "Returns, not price levels, are stationary — always check units before fitting a model.",
    "A high R² between two trending series is the classic warning sign of a spurious regression.",
    "Lag every predictor by one day, or your 'forecast' quietly contains the answer.",
    "Student-t innovations matter: with ν≈3, tail risk is far larger than a normal model suggests.",
    "Persistence close to 1 in GARCH means volatility shocks fade slowly (long memory in risk).",
    "Regime labels that last only a few days are usually fat-tail mixtures, not structural regimes.",
    "VIF above 10 flags overlapping predictors; fixing it rarely creates predictive power by itself.",
    "Standardise before PCA — otherwise the variable with the biggest units 'explains' everything.",
    "Out-of-sample error is the honest scoreboard; in-sample R² only measures how well you fitted noise.",
    "Cointegration needs two I(1) series that share a trend; BTC price vs volume does not qualify.",
    "A Value-at-Risk number is a threshold that is exceeded (1 − confidence) of the time, not a maximum loss.",
    "Check the data first: one bad volume print can create a −110,000 residual.",
]


def daily_tip(today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    return TIPS[today.toordinal() % len(TIPS)]
