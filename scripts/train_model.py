"""Estimate the models offline and freeze their parameters into model/model.pkl.

Run:  python scripts/train_model.py
The Streamlit app only *applies* these parameters (utils/engine.py); if the file is
missing it falls back to the built-in copy in utils/model_store.py.
"""
from __future__ import annotations

import pathlib, sys, datetime as dt
import numpy as np
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, features as ft, regime as rg          # noqa: E402
from utils import engine                                     # noqa: E402

data.quiet()
df, prov = data.get_data("BTC-USD", "cache")
d = data.prepare(df, "BTC-USD")
r = d["ret_pct"]

g = ft.garch_variance(r, "GARCH(1,1)-t")
p = g["params"]
mk = rg.fit_markov(r)
hi = mk["high_vol_regime"]; lo = 1 - hi
reg = {x["regime"]: x for x in mk["regimes"]}
P = np.asarray(mk["transition"], float)

bundle = dict(
    name="BTC Econometrics Lab - fitted parameters", version="2.1.0",
    trained_on=prov, window=[str(r.index[0].date()), str(r.index[-1].date())], n_obs=int(len(r)),
    trained_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    garch=dict(spec=g["spec"], mu=p["mu"], omega=p["omega"], alpha=p["alpha[1]"], beta=p["beta[1]"],
               nu=p["nu"], persistence=g["persistence"], at_boundary=g["at_boundary"]),
    markov=dict(mu=[reg[0]["mu"], reg[1]["mu"]], sigma2=[reg[0]["sigma2"], reg[1]["sigma2"]],
                P=P.tolist(), turbulent=1, quality=mk["quality"],
                expected_duration=[reg[0]["expected_duration"], reg[1]["expected_duration"]]),
    sample_var=float(r.var()),
    train_cond_vol=[round(float(x), 4) for x in g["cond_vol"].values[::5]],   # thinned, for percentiles
)

# ---- validate the lightweight filters against the estimation libraries ----------
var, _ = engine.garch_filter(r.values, p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"], float(r.var()))
gap_vol = float(np.max(np.abs(np.sqrt(var[100:]) - g["cond_vol"].values[100:])))
pr = engine.markov_filter(r.values, reg_mu := bundle["markov"]["mu"], bundle["markov"]["sigma2"], P)
gap_p = float(np.max(np.abs(pr[:, 1] - mk["filtered"].iloc[:, hi].values)))
bundle["validation"] = dict(max_abs_gap_cond_vol_pct=gap_vol, max_abs_gap_filtered_prob=gap_p)
print("validation:", bundle["validation"])
assert gap_vol < 0.05 and gap_p < 0.02, "lightweight filter disagrees with the estimation library"

(ROOT / "model").mkdir(exist_ok=True)
joblib.dump(bundle, ROOT / "model" / "model.pkl", compress=3)
print("saved model/model.pkl |", {k: round(v, 5) if isinstance(v, float) else v for k, v in bundle["garch"].items()})
print("markov:", bundle["markov"])
