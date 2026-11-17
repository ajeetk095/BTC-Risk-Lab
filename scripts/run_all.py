"""Headless end-to-end run: reproduces every result and writes figures + JSON."""
from __future__ import annotations
import json, pathlib, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import data, stationarity as st, regime as rg, collinearity as mc, features as ft
data.quiet(); warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT/"figures"; FIG.mkdir(exist_ok=True)
OUT = ROOT/"report"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.grid":True,"grid.alpha":.3,
                     "axes.spines.top":False,"axes.spines.right":False,"font.size":9})
R={}

df,prov = data.get_data("BTC-USD","cache"); d = data.prepare(df,"BTC-USD")
R["provenance"]=prov; R["n_obs"]=len(d); R["volume_outliers"]=[str(i.date()) for i in d.index[d["vol_outlier"]]]; R["window"]=[str(d.index[0].date()),str(d.index[-1].date())]
print(f"== {prov} | n={len(d)} | {R['window'][0]} .. {R['window'][1]}")

# 1 stationarity
ur = st.unit_root_table(d,["Close","lnClose","Volume","lnVolume","dClose","ret_pct","dVol_bn"])
print("\n-- unit roots --\n", ur.to_string(index=False))
R["unit_roots"]=ur.to_dict("records")
eg = st.engle_granger(d,"Close","Volume")
R["eg_btc_volume"]={k:eg[k] for k in ["beta","r2","dw","naive_adf_p","eg_stat","eg_p","eg_crit","cointegrated","precondition_ok","condition_number","n"]}
R["eg_orders"]=eg["orders"]
print("\n-- EG BTC/Volume --", {k:round(v,5) if isinstance(v,float) else v for k,v in R["eg_btc_volume"].items() if k!='eg_crit'})
print("   orders:", [(o['series'],o['order']) for o in eg['orders']])
ecm_bad = st.fit_ecm(eg,d,"Close","Volume")
R["ecm_btc_volume"]={k:ecm_bad[k] for k in ["phi","phi_p","gamma","r2","valid","n"]}
print("   ECM phi=",round(ecm_bad['phi'],6),"p=",round(ecm_bad['phi_p'],4),"valid=",ecm_bad['valid'])

var = st.fit_var(d,["dClose","dVol_bn"],"bic")
R["var"]={"maxlags":var["maxlags"],"orders":var["orders"],"p":var["p"],"n":var["n"],
          "stable":var["stable"],"whiteness_p":var["whiteness_p"],"aic":var["aic"],"bic":var["bic"],"granger":var["granger"]}
print("\n-- VAR --", R["var"])

# valid equilibrium pair
g1,_=data.get_data("GOOGL"); g2,_=data.get_data("GOOG")
p1,p2=data.prepare(g1,"GOOGL"),data.prepare(g2,"GOOG")
pair=pd.concat([p1["Close"].rename("GOOGL"),p2["Close"].rename("GOOG")],axis=1).dropna()
eg2=st.engle_granger(pair,"GOOGL","GOOG",log=True)
ecm2=st.fit_ecm(eg2,pair,"GOOGL","GOOG",log=True)
R["eg_googl_goog"]={k:eg2[k] for k in ["beta","r2","eg_p","cointegrated","precondition_ok","n"]}
R["ecm_googl_goog"]={k:ecm2[k] for k in ["phi","phi_p","gamma","r2","half_life_days","valid","n"]}
print("\n-- EG GOOGL/GOOG (log) --",R["eg_googl_goog"]); print("   ECM:",R["ecm_googl_goog"])

fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
ax[0].plot(d.index,d["Close"],lw=.8); ax[0].set_ylabel("BTC-USD close ($)"); ax[0].set_title("Level: I(1) - non-stationary")
ax[1].plot(d.index,d["ret_pct"],lw=.6,color="darkorange"); ax[1].set_ylabel("Daily return (%)"); ax[1].set_title("First difference / return: I(0) - stationary")
ax[1].set_xlabel("Date"); fig.tight_layout(); fig.savefig(FIG/"01_stationarity.png"); plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].scatter(d["Volume"]/1e9,d["Close"],s=5,alpha=.4); ax[0].set_xlabel("Volume (bn)"); ax[0].set_ylabel("Close ($)")
ax[0].set_title(f"Spurious regression: R2={eg['r2']:.3f}, DW={eg['dw']:.3f}")
ax[1].plot(eg["resid"].index,eg["resid"].values,lw=.7,color="crimson"); ax[1].axhline(0,color="k",lw=.8)
ax[1].set_title(f"EG residuals: ADF-like p={eg['eg_p']:.3f} -> unit root"); ax[1].set_xlabel("Date"); ax[1].set_ylabel("Residual ($)")
fig.tight_layout(); fig.savefig(FIG/"02_spurious.png"); plt.close(fig)

fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
ax[0].plot(eg2["resid"].index,eg2["resid"].values,lw=.8,color="seagreen"); ax[0].axhline(0,color="k",lw=.8)
ax[0].set_title(f"Cointegrated pair ln(GOOGL)~ln(GOOG): beta={eg2['beta']:.4f}, EG p={eg2['eg_p']:.4f}")
ax[0].set_ylabel("Equilibrium gap (log)")
ax[1].plot(ecm2["data"].index,ecm2["data"]["ec_lag"],lw=.7,label="lagged gap")
ax[1].plot(ecm2["data"].index,ecm2["data"]["dY"],lw=.7,alpha=.7,label="d ln GOOGL")
ax[1].set_title(f"ECM: phi={ecm2['phi']:.4f} (p={ecm2['phi_p']:.1e}), half-life={ecm2['half_life_days']:.1f} days")
ax[1].legend(fontsize=7); ax[1].set_xlabel("Date"); fig.tight_layout(); fig.savefig(FIG/"03_ecm.png"); plt.close(fig)

irf=var["result"].irf(12)
fig=irf.plot(orth=True,figsize=(9,5)); fig.suptitle(f"VAR({var['p']}) orthogonalised impulse responses",y=1.02)
fig.savefig(FIG/"04_var_irf.png",bbox_inches="tight"); plt.close(fig)

# 2 regime - recommended spec (percent returns) vs original spec (dollar diffs)
mk = rg.fit_markov(d["ret_pct"])
mk_dollar = rg.fit_markov(d["dClose"])
def _pack(m):
    return {"loglik":m["loglik"],"aic":m["aic"],"bic":m["bic"],"starts":m["starts_tried"],
            "regimes":m["regimes"],"transition":m["transition"].tolist(),
            "n_switches":m["n_switches"],"min_duration":m["min_duration"],"quality":m["quality"],"predict_range":m["predict_range"],
            "high_vol_regime":m["high_vol_regime"]}
def _byyear(m):
    lab=m["label"]; hi=m["high_vol_regime"]
    s=pd.Series((lab==hi).astype(int).values,index=lab.index)
    return {int(k):float(v) for k,v in s.groupby(s.index.year).mean().items()}
R["markov_returns"]=_pack(mk); R["markov_dollar"]=_pack(mk_dollar)
R["highvol_share_by_year"]={"returns_spec":_byyear(mk),"dollar_spec":_byyear(mk_dollar)}
print("\n-- Markov on PERCENT RETURNS (recommended) --")
for r_ in mk["regimes"]: print("   ",{k:(round(v,4) if isinstance(v,float) else v) for k,v in r_.items()})
print("-- Markov on DOLLAR DIFFERENCES (original spec) -- predict() range =",
      [round(x,2) for x in mk_dollar["predict_range"]],"-> DOLLARS, not a probability")
print("   high-vol share by year, dollar spec :",{k:round(v,2) for k,v in _byyear(mk_dollar).items()})
print("   high-vol share by year, returns spec:",{k:round(v,2) for k,v in _byyear(mk).items()})
ep=rg.regime_episodes(mk["label"],min_len=3); print("   episodes>=3d:",len(ep))
R["episodes"]=ep.head(15).to_dict("records")

fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
for ax,(m,ttl,col) in zip(axs,[(mk,"Recommended: percent returns","seagreen"),
                               (mk_dollar,"Original: dollar differences (price-level artefact)","crimson")]):
    ax.plot(m["series"].index,m["series"].values,lw=.5,color="0.55",label="Series")
    ax.set_ylabel("Return (%)" if m is mk else "Price change ($)")
    ax2=ax.twinx(); ax2.fill_between(m["prob_high"].index,0,m["prob_high"].values,color=col,alpha=.4)
    ax2.set_ylim(0,1); ax2.set_ylabel("P(turbulent)"); ax2.grid(False)
    ax.set_title(ttl,fontsize=10)
axs[1].set_xlabel("Date")
fig.suptitle("Smoothed regime probabilities - specification matters",y=.995)
fig.tight_layout(); fig.savefig(FIG/"05_regime.png"); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4))
yrs=sorted(R["highvol_share_by_year"]["returns_spec"])
a=[R["highvol_share_by_year"]["dollar_spec"][y] for y in yrs]
b=[R["highvol_share_by_year"]["returns_spec"][y] for y in yrs]
w=.38; xx=np.arange(len(yrs))
ax.bar(xx-w/2,a,w,label="Dollar differences (original)",color="crimson")
ax.bar(xx+w/2,b,w,label="Percent returns (corrected)",color="seagreen")
ax.set_xticks(xx); ax.set_xticklabels(yrs); ax.set_ylabel("Share of days labelled turbulent")
ax.set_title("Dollar-based regimes track the price level, not risk")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"05b_regime_artefact.png"); plt.close(fig)

# 3 multicollinearity
X,y,_=mc.build_features(d); vif=mc.vif_table(X); pca=mc.pca_analysis(X); rem=mc.remedy_comparison(X,y)
R["vif"]=vif.to_dict("records"); R["pca"]={"evr":pca["evr"].tolist(),"n_for_90":pca["n_for_90"],
   "cond_raw":pca["cond_raw"],"cond_scaled":pca["cond_scaled"]}
R["remedies"]={"mse":rem["mse"],"gain_pct":rem["gain_pct"],"dropped":rem["dropped"],
   "ridge_alpha":rem["ridge_alpha"],"ridge_at_edge":rem["ridge_at_edge"],"folds_better":rem["folds_better"],"fold_mse":rem["fold_mse"],"ols_r2":float(rem["ols"].rsquared),"n_sig":rem["n_sig"],"n_pred":rem["n_pred"],"n":len(y)}
print("\n-- VIF --\n",vif.to_string(index=False))
print("   PCA cum:",np.round(pca["cum"],3),"n_for_90:",pca["n_for_90"])
print("   OOS MSE:",{k:round(v,4) for k,v in rem["mse"].items()})
print("   gain %:",{k:round(v,2) for k,v in rem["gain_pct"].items()},"dropped:",rem["dropped"])

fig,ax=plt.subplots(1,2,figsize=(11,4))
cols=["crimson" if f=="severe" else ("orange" if f=="moderate" else "seagreen") for f in vif["flag"]]
ax[0].barh(vif["feature"],vif["VIF"],color=cols); ax[0].axvline(10,color="k",ls="--",lw=1)
ax[0].invert_yaxis(); ax[0].set_xlabel("Variance Inflation Factor"); ax[0].set_title("Real multicollinearity (VIF > 10 = severe)")
ax[1].bar(range(1,len(pca["evr"])+1),pca["evr"],color="steelblue",alpha=.8,label="individual")
ax[1].plot(range(1,len(pca["cum"])+1),pca["cum"],"o-",color="crimson",label="cumulative")
ax[1].axhline(.9,color="k",ls="--",lw=1); ax[1].set_xlabel("Principal component"); ax[1].set_ylabel("Explained variance ratio")
ax[1].set_title(f"Standardised PCA: {pca['n_for_90']} PCs reach 90%"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"06_collinearity.png"); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4))
names=list(rem["mse"]); vals=[rem["mse"][k] for k in names]
ax.bar(names,vals,color=["crimson","seagreen","steelblue","orange"])
for i,v in enumerate(vals): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
ax.set_ylabel("Out-of-sample MSE (expanding-window CV)"); ax.set_title("Remedies vs OLS: out-of-sample differences are small")
fig.tight_layout(); fig.savefig(FIG/"07_remedies.png"); plt.close(fig)

# 4 features
fp=ft.feature_panel(d["ret_pct"]); cmp=ft.compare_specs(d["ret_pct"]); fc=ft.forecast_risk(fp["garch"],10)
g=fp["garch"]
R["features"]={"adf_stat":float(fp["adf"][0]),"adf_p":float(fp["adf"][1]),
  "arma_params":{k:float(v) for k,v in fp["arma"]["result"].params.items()},
  "garch_spec":g["spec"],"garch_params":g["params"],"persistence":g["persistence"],
  "integrated":g["integrated"],"at_boundary":g["at_boundary"],"half_life_days":g["half_life_days"],"nu":g["nu"],
  "unconditional_vol":g["unconditional_vol"],"kalman":{k:fp["kalman"][k] for k in ["sigma2_level","sigma2_irregular","signal_to_noise"]},
  "sq_vs_garch_corr":fp["sq_vs_garch_corr"],"spec_table":cmp.to_dict("records")}
R["risk_forecast"]=fc.to_dict("records")
print("\n-- GARCH --",g["spec"],{k:round(v,5) for k,v in g["params"].items()})
print("   persistence=",round(g["persistence"],5),"IGARCH=",g["integrated"],"at_boundary=",g["at_boundary"],"nu=",round(g["nu"],3))
print("   spec comparison:\n",cmp.to_string(index=False))
print("   Kalman s/n=",fp["kalman"]["signal_to_noise"])
print("   corr(r2, cond var)=",round(fp["sq_vs_garch_corr"],4))
print("   volume outliers cleaned:",R["volume_outliers"],"| ridge_at_edge:",rem["ridge_at_edge"],"| regime quality:",mk["quality"])
print("   VaR table:\n",fc.head(3).to_string(index=False))

fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True)
P=fp["panel"]
axs[0].plot(P.index,P["return_pct"],lw=.5,color="0.6",label="Return (%)")
axs[0].plot(P.index,P["conditional_mean"],lw=1,color="navy",label=f"Conditional mean ARMA{fp['arma']['order']}")
axs[0].set_ylabel("Mean (%)"); axs[0].legend(fontsize=7,loc="upper left")
axs[1].plot(P.index,P["conditional_variance"],lw=.9,color="darkorange",label=f"Conditional variance {g['spec']}")
axs[1].set_ylabel("Variance (%²)"); axs[1].legend(fontsize=7,loc="upper left")
axs[2].plot(P.index,P["squared_return"],lw=.5,color="crimson",label="Squared return (noisy proxy)")
axs[2].set_ylabel("Squared return"); axs[2].legend(fontsize=7,loc="upper left")
axs[3].plot(P.index,P["return_pct"],lw=.5,color="0.7",alpha=.7,label="Return (%)")
axs[3].plot(P.index,P["kalman_level"],lw=1.4,color="green",label="Kalman smoothed local level")
axs[3].set_ylabel("Level (%)"); axs[3].set_xlabel("Date"); axs[3].legend(fontsize=7,loc="upper left")
fig.suptitle("Extracted features - BTC daily percent returns",y=.995)
fig.tight_layout(); fig.savefig(FIG/"08_features.png"); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4))
ax.plot(fc["horizon_days"],fc["vol_forecast_pct"],"o-",label="Volatility forecast")
ax.plot(fc["horizon_days"],-fc["VaR95_pct"],"s--",label="95% VaR (loss, %)")
ax.plot(fc["horizon_days"],-fc["VaR99_pct"],"^--",label="99% VaR (loss, %)")
ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Percent"); ax.legend(fontsize=8)
ax.set_title(f"Deployable risk output - {g['spec']}, persistence={g['persistence']:.3f}")
fig.tight_layout(); fig.savefig(FIG/"09_risk.png"); plt.close(fig)

fp["panel"].to_csv(OUT/"feature_panel.csv")
mk["smoothed"].assign(regime=mk["label"]).to_csv(OUT/"regime_probabilities.csv")
vif.to_csv(OUT/"vif_table.csv",index=False); ur.to_csv(OUT/"unit_root_tests.csv",index=False)
def enc(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.ndarray,)): return o.tolist()
    if isinstance(o,(pd.Timestamp,)): return str(o)
    return str(o)
(OUT/"results.json").write_text(json.dumps(R,indent=2,default=enc))
print("\nSaved figures ->",FIG," results ->",OUT/"results.json")
