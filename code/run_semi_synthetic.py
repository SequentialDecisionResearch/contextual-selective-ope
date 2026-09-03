"""BC-SPI Semi-Synthetic benchmark: real X/Y + simulated bandit logging.

Primary datasets: Digits, Wine, Breast Cancer.  The policy-training split is
independent from OPE evaluation.  Natural and deliberately local-stressed
candidates are evaluated under paired good/poor overlap logging regimes.

Run directly in Spyder. The working directory is forced to the BC-SPI project root,
and outputs are saved under the fixed bcspi_results directory.




"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code")
if not PROJECT_DIR.exists():
    raise FileNotFoundError(f"BC-SPI project directory does not exist: {PROJECT_DIR}")
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

print("Python version :", sys.version)
print("Python exe     :", sys.executable)
print("Working dir    :", os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, load_wine, load_breast_cancer, fetch_covtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

import bcspi_config as cfg
from bcspi_core import (
    BanditData, validate_bandit_data, save_csv, save_json, save_show, print_frame,
    plot_calibration, plot_safety_coverage, monte_carlo_summary,
)
from bcspi_experiment import evaluate_generic_bandit


POLICY_TRAIN_FRACTION = 0.40
PI0_SMOOTHING = 0.08
PI1_SMOOTHING = 0.05
LOCAL_STRESS_RHO = 0.40
DATASETS = ["digits", "wine", "breast_cancer"] + (["covtype"] if cfg.INCLUDE_COVTYPE else [])

LOGGING_REGIMES = {
    "good_overlap": dict(kind="mixture", uniform_mix=0.50, mismatch=0.00),
    "poor_overlap": dict(kind="temperature", tau=0.25, exploration_floor=0.01, mismatch=0.55),
}
GROUP_SCHEMES = {
    "10_20_70": (0.10, 0.30),
    "15_25_60": (0.15, 0.40),
    "20_30_50": (0.20, 0.50),
}

# Compute-efficient PAPER protocol.  These values reduce repeated Monte Carlo work
# while preserving the primary design, paired logging regimes, both candidate
# variants, both nuisance models in the fixed-population phase, and outer-split
# robustness.  Smoke mode continues to use the values from bcspi_config.py.
PAPER_PRIMARY_FIXED_LOG_DRAWS = 250
PAPER_SENSITIVITY_FIXED_LOG_DRAWS = 100
PAPER_OUTER_SPLITS = 20
PAPER_OUTER_LOG_DRAWS = 10


def softmax(z):
    z=np.asarray(z,float); z-=z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)


def smooth_probs(p, eps):
    p=np.asarray(p,float); k=p.shape[1]; out=(1-eps)*p+eps/k; return out/out.sum(axis=1,keepdims=True)


def behavior_policy_from_pi0(pi0, rcfg):
    n,k=pi0.shape; uniform=np.full((n,k),1/k,float); mismatch=float(rcfg.get("mismatch",0))
    source=(1-mismatch)*pi0+mismatch*np.roll(pi0,1,axis=1)
    if rcfg["kind"]=="mixture":
        mix=float(rcfg["uniform_mix"]); mu=(1-mix)*source+mix*uniform
    else:
        tau=float(rcfg["tau"]); floor=float(rcfg["exploration_floor"])
        mu=softmax(np.log(np.clip(source,1e-12,1))/tau); mu=(1-floor)*mu+floor*uniform
    return mu/mu.sum(axis=1,keepdims=True)


def sample_with_uniform(prob, u):
    cdf=np.cumsum(prob,axis=1); cdf[:,-1]=1.0
    return (np.asarray(u)[:,None] > cdf).sum(axis=1).astype(int)


def load_raw(name):
    if name=="digits": bunch=load_digits()
    elif name=="wine": bunch=load_wine()
    elif name=="breast_cancer": bunch=load_breast_cancer()
    elif name=="covtype": bunch=fetch_covtype(as_frame=False)
    else: raise ValueError(name)
    X=np.asarray(bunch.data,float); enc=LabelEncoder(); y=enc.fit_transform(np.asarray(bunch.target)).astype(int)
    if name=="covtype" and cfg.COVTYPE_MAX_ROWS and len(y)>cfg.COVTYPE_MAX_ROWS:
        X,_,y,_=train_test_split(X,y,train_size=cfg.COVTYPE_MAX_ROWS,random_state=cfg.RANDOM_SEED,stratify=y)
    return X,y


def train_policies(Xtr,ytr,seed):
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,solver="lbfgs"))
    candidate=HistGradientBoostingClassifier(learning_rate=.08,max_iter=250,max_leaf_nodes=31,l2_regularization=.10,random_state=seed)
    baseline.fit(Xtr,ytr);candidate.fit(Xtr,ytr);return baseline,candidate


def confidence_groups(pi0,cuts=(.10,.30)):
    conf=pi0.max(axis=1); q=np.quantile(conf,cuts); return np.digitize(conf,q,right=True).astype(int),q


def local_stress_candidate(pi1,groups,rho=LOCAL_STRESS_RHO):
    out=pi1.copy();m=groups==0
    if m.any(): out[m]=(1-rho)*pi1[m]+rho*np.roll(pi1[m],1,axis=1)
    return out/out.sum(axis=1,keepdims=True)


def exact_truth(y,pi0,pi1,groups):
    idx=np.arange(len(y));v0=pi0[idx,y];v1=pi1[idx,y];d=v1-v0;rows=[]
    for g in range(3):
        m=groups==g
        rows.append(dict(group=g,n=int(m.sum()),population_weight=float(m.mean()),v0_true=float(v0[m].mean()),v1_true=float(v1[m].mean()),delta_true=float(d[m].mean()),fraction_pointwise_delta_negative=float((d[m]<0).mean())))
    rows.append(dict(group=-1,n=len(y),population_weight=1.,v0_true=float(v0.mean()),v1_true=float(v1.mean()),delta_true=float(d.mean()),fraction_pointwise_delta_negative=float((d<0).mean())))
    return pd.DataFrame(rows)


def prepare_population(name,outer_seed,group_scheme="10_20_70"):
    X,y=load_raw(name)
    Xtr,Xev,ytr,yev=train_test_split(X,y,train_size=POLICY_TRAIN_FRACTION,random_state=outer_seed,stratify=y)
    b,c=train_policies(Xtr,ytr,outer_seed)
    pi0=smooth_probs(b.predict_proba(Xev),PI0_SMOOTHING)
    pi1nat=smooth_probs(c.predict_proba(Xev),PI1_SMOOTHING)
    groups,cuts=confidence_groups(pi0,GROUP_SCHEMES[group_scheme])
    pi1stress=local_stress_candidate(pi1nat,groups)
    return Xev,yev,pi0,{"natural":pi1nat,"local_stress":pi1stress},groups,cuts


def generate_logged(X,y,pi0,pi1,groups,regime,u):
    mu=behavior_policy_from_pi0(pi0,LOGGING_REGIMES[regime]);a=sample_with_uniform(mu,u)
    p=mu[np.arange(len(y)),a];r=(a==y).astype(float)
    data=BanditData(X=X,action=a,reward=r,pscore=p,pi0=pi0,pi1=pi1,group=groups,sample_id=np.arange(len(y)))
    validate_bandit_data(data,f"semi/{regime}")
    return data


def _plot_outputs(gates,decisions,calibration):
    out=cfg.FIGURES_DIR/"semi_synthetic";out.mkdir(parents=True,exist_ok=True)
    # Good vs poor overlap paired evidence.
    s=gates[(gates.method=="bb_dr") & (gates.group>=0) & (gates.group_scheme=="10_20_70")]
    if len(s):
        ss=s.groupby(["dataset","candidate_variant","regime"],as_index=False).agg(ess_fraction=("ess_fraction","mean"),interval_width=("interval_width","mean"))
        save_csv(ss,cfg.TABLES_DIR/"semi_good_poor_overlap_summary.csv")
        fig,ax=plt.subplots(figsize=(8,5))
        for key,d in ss.groupby(["dataset","candidate_variant"]):
            d=d.set_index("regime").reindex(["good_overlap","poor_overlap"]).reset_index()
            ax.plot(d.regime,d.interval_width,marker="o",label=f"{key[0]}-{key[1]}")
        ax.set_ylabel("Mean 95% interval width");ax.set_title("Semi-Synthetic: same policy truth, different evaluability");ax.legend(fontsize=7)
        save_show(fig,out/"fig_semi_good_vs_poor_width.png")

    # Decision quality across datasets / variants.
    s=decisions[(decisions.method=="bb_dr") & (decisions.group_scheme=="10_20_70")]
    if len(s):
        ss=s.groupby(["dataset","candidate_variant","regime"],as_index=False)[["deployment_coverage","harmful_exposure","oracle_regret"]].mean()
        fig,ax=plt.subplots(figsize=(9,5))
        x=np.arange(len(ss));ax.bar(x,ss.harmful_exposure);ax.set_xticks(x);ax.set_xticklabels((ss.dataset+"\n"+ss.candidate_variant+"\n"+ss.regime),rotation=45,ha="right",fontsize=7)
        ax.set_ylabel("Harmful exposure");ax.set_title("Semi-Synthetic decision harm")
        save_show(fig,out/"fig_semi_decision_harm.png")

    c=calibration[(calibration.group>=0)&(calibration.group_scheme=="10_20_70")]
    if len(c):
        cc=c.groupby(["method","nominal"],as_index=False).covered.mean().rename(columns={"covered":"empirical"})
        plot_calibration(cc,"Semi-Synthetic interval calibration",out/"fig_semi_calibration.png")


def run_semi_synthetic():
    cfg.print_protocol();print("\n[RUN] Semi-Synthetic real-X/Y benchmark\n")
    quick=os.environ.get("BCSPI_INTERNAL_QUICK_TEST","0")=="1"
    point_all=[];gate_all=[];dec_all=[];cal_all=[];front_all=[];diag_all=[];truth_all=[]

    # Primary fixed-population repeated logging: one frozen train/eval split per dataset.
    datasets=DATASETS[:1] if quick else DATASETS
    for di,name in enumerate(datasets):
        outer_seed=cfg.RANDOM_SEED+10_000*di
        schemes=["10_20_70"]
        if cfg.RUN_MODE=="paper" and not quick: schemes=list(GROUP_SCHEMES)
        for group_scheme in schemes:
            X,y,pi0,cands,groups,cuts=prepare_population(name,outer_seed,group_scheme)
            variants=["natural","local_stress"] if not quick else ["local_stress"]
            if quick:
                n_draws = 1
            elif cfg.RUN_MODE == "paper":
                n_draws = (
                    PAPER_PRIMARY_FIXED_LOG_DRAWS
                    if group_scheme == "10_20_70"
                    else PAPER_SENSITIVITY_FIXED_LOG_DRAWS
                )
            else:
                n_draws = cfg.SEMI_FIXED_LOG_DRAWS
            for draw in range(n_draws):
                rng=np.random.default_rng(outer_seed+1000+draw);u=rng.random(len(y))
                for variant in variants:
                    pi1=cands[variant];truth=exact_truth(y,pi0,pi1,groups)
                    truth_tmp=truth.copy();truth_tmp["dataset"]=name;truth_tmp["candidate_variant"]=variant;truth_tmp["group_scheme"]=group_scheme;truth_tmp["outer_seed"]=outer_seed
                    truth_all.append(truth_tmp)
                    for regime in ["good_overlap","poor_overlap"]:
                        print(f"[Semi fixed] {name} {variant} {group_scheme} {regime} draw={draw+1}/{n_draws}")
                        data=generate_logged(X,y,pi0,pi1,groups,regime,u)
                        models=cfg.SEMI_NUISANCE_MODELS if not quick else ["linear"]
                        for model in models:
                            meta={"layer":"semi","phase":"fixed_population","dataset":name,"candidate_variant":variant,"regime":regime,"group_scheme":group_scheme,"outer_seed":outer_seed,"logging_draw":draw,"n":len(y)}
                            ev=evaluate_generic_bandit(data,truth,meta,model,outer_seed+draw)
                            point_all.append(ev.point);gate_all.append(ev.gates);dec_all.append(ev.decisions);cal_all.append(ev.calibration);front_all.append(ev.frontiers);diag_all.append(ev.diagnostics)

    # Outer-split robustness: primary group scheme only.  It asks whether the
    # conclusion depends on one classifier train/eval split.
    if not quick:
        outer_splits = PAPER_OUTER_SPLITS if cfg.RUN_MODE == "paper" else cfg.SEMI_OUTER_SPLITS
        outer_log_draws = PAPER_OUTER_LOG_DRAWS if cfg.RUN_MODE == "paper" else cfg.SEMI_OUTER_LOG_DRAWS
        for di,name in enumerate(DATASETS):
            for outer in range(outer_splits):
                outer_seed=cfg.RANDOM_SEED+5_000_000+di*100_000+outer
                X,y,pi0,cands,groups,cuts=prepare_population(name,outer_seed,"10_20_70")
                for draw in range(outer_log_draws):
                    u=np.random.default_rng(outer_seed+draw+999).random(len(y))
                    for variant,pi1 in cands.items():
                        truth=exact_truth(y,pi0,pi1,groups)
                        for regime in ["good_overlap","poor_overlap"]:
                            data=generate_logged(X,y,pi0,pi1,groups,regime,u)
                            # Outer robustness uses the main nuisance model to control compute.
                            model="linear"
                            meta={"layer":"semi","phase":"outer_robustness","dataset":name,"candidate_variant":variant,"regime":regime,"group_scheme":"10_20_70","outer_seed":outer_seed,"logging_draw":draw,"n":len(y)}
                            ev=evaluate_generic_bandit(data,truth,meta,model,outer_seed+draw)
                            point_all.append(ev.point);gate_all.append(ev.gates);dec_all.append(ev.decisions);cal_all.append(ev.calibration);front_all.append(ev.frontiers);diag_all.append(ev.diagnostics)

    point=pd.concat(point_all,ignore_index=True);gates=pd.concat(gate_all,ignore_index=True);decisions=pd.concat(dec_all,ignore_index=True)
    calibration=pd.concat(cal_all,ignore_index=True);frontiers=pd.concat(front_all,ignore_index=True);diagnostics=pd.concat(diag_all,ignore_index=True)
    truths=pd.concat(truth_all,ignore_index=True).drop_duplicates() if truth_all else pd.DataFrame()

    save_csv(point,cfg.TABLES_DIR/"semi_point_estimates_long.csv")
    save_csv(gates,cfg.TABLES_DIR/"semi_gate_results_long.csv")
    save_csv(decisions,cfg.TABLES_DIR/"semi_decision_results.csv")
    save_csv(calibration,cfg.TABLES_DIR/"semi_calibration_long.csv")
    save_csv(frontiers,cfg.TABLES_DIR/"semi_safety_coverage_frontiers.csv")
    save_csv(diagnostics,cfg.TABLES_DIR/"semi_evidence_diagnostics.csv")
    if len(truths): save_csv(truths,cfg.TABLES_DIR/"semi_evaluation_only_truth_summary.csv")

    dec_summary=monte_carlo_summary(decisions,["phase","dataset","candidate_variant","regime","method"],["deployment_coverage","harmful_exposure","selective_value","oracle_regret"])
    cal_summary=calibration.groupby(["phase","dataset","candidate_variant","regime","method","nominal"],as_index=False).covered.mean().rename(columns={"covered":"empirical"})
    save_csv(dec_summary,cfg.TABLES_DIR/"semi_decision_summary.csv");save_csv(cal_summary,cfg.TABLES_DIR/"semi_calibration_summary.csv")
    _plot_outputs(gates,decisions,calibration)

    print_frame("Semi-Synthetic decision summary",dec_summary,50)
    paired=gates[(gates.phase=="fixed_population")&(gates.method=="bb_dr")&(gates.group>=0)&(gates.group_scheme=="10_20_70")]
    if len(paired):
        p=paired.groupby(["dataset","candidate_variant","regime"],as_index=False).agg(ess=("ess_fraction","mean"),width=("interval_width","mean"))
        print_frame("Good vs poor overlap evidence",p,30)

    if quick:
        manifest_primary_draws = 1
        manifest_sensitivity_draws = 1
        manifest_outer_splits = 0
        manifest_outer_log_draws = 0
    elif cfg.RUN_MODE == "paper":
        manifest_primary_draws = PAPER_PRIMARY_FIXED_LOG_DRAWS
        manifest_sensitivity_draws = PAPER_SENSITIVITY_FIXED_LOG_DRAWS
        manifest_outer_splits = PAPER_OUTER_SPLITS
        manifest_outer_log_draws = PAPER_OUTER_LOG_DRAWS
    else:
        manifest_primary_draws = cfg.SEMI_FIXED_LOG_DRAWS
        manifest_sensitivity_draws = cfg.SEMI_FIXED_LOG_DRAWS
        manifest_outer_splits = cfg.SEMI_OUTER_SPLITS
        manifest_outer_log_draws = cfg.SEMI_OUTER_LOG_DRAWS
    save_json({
        "run_mode": cfg.RUN_MODE,
        "datasets": datasets,
        "fixed_logging_draws": manifest_primary_draws,
        "primary_fixed_logging_draws": manifest_primary_draws,
        "sensitivity_fixed_logging_draws": manifest_sensitivity_draws,
        "outer_splits": manifest_outer_splits,
        "outer_logging_draws": manifest_outer_log_draws,
        "compute_efficient_paper_protocol": bool(cfg.RUN_MODE == "paper" and not quick),
    }, cfg.RESULTS_DIR/"semi_run_manifest.json")
    print("\nSemi-Synthetic run completed without fatal validation errors.")
    print(f"Tables : {cfg.TABLES_DIR}")
    print(f"Figures: {cfg.FIGURES_DIR/'semi_synthetic'}")
    return point,gates,decisions,calibration,frontiers,diagnostics


if __name__=="__main__":
    run_semi_synthetic()
