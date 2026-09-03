"""BC-SPI Controlled Synthetic experiments (S0--S6).

Spyder usage
------------
1. Keep bcspi_config.py in RUN_MODE="paper" for the formal rerun.
2. Run this file directly in Spyder (F5) from C:\\study_notes\\traval_rec\\Contextual_bandit_bayesian_ope_safe_imp\\BCSPI_python_code.
3. Final CSV/PNG outputs overwrite only the Synthetic outputs under
   C:\\study_notes\\traval_rec\\Contextual_bandit_bayesian_ope_safe_imp\\BCSPI_python_code\\bcspi_results; Semi/OBD outputs are untouched.
4. Expensive work is checkpointed in cache/synthetic_paper_checkpoint_efficient_v2 and resumes
   automatically after interruption when the protocol signature is unchanged.

This runner implements the research protocol in the revised memo:
S0 calibration, S1 overlap, S2 rarity/pooling, S3 covariate/composition shift,
S4 reward-model misspecification, S5 frozen combined stress, and a deliberately
smaller S6 temporal/historical-relevance robustness layer.  The formal paper
protocol caps logged n at 10,000; n=20,000 is intentionally not used.
"""
from __future__ import annotations

import math
import os
import json
import hashlib
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp

import bcspi_config as cfg
from bcspi_core import (
    BanditData,
    validate_bandit_data,
    dr_uncertainty_by_group,
    summaries_from_draw_dict,
    decision_metrics_from_truth,
    safety_coverage_frontier_from_draws,
    safe_coverage_at_eta,
    empirical_bayes_partial_pooling,
    complete_pooling_draws,
    estimate_density_ratio_log_to_deploy,
    recency_weights,
    weighted_mean,
    evidence_diagnostics,
    save_csv,
    save_json,
    save_show,
    print_frame,
    plot_calibration,
    plot_safety_coverage,
    monte_carlo_summary,
)
from bcspi_experiment import evaluate_generic_bandit


# =============================================================================
# Frozen DGP
# =============================================================================
D = 10
K = 5
G = 3
PI0_ORACLE_WEIGHT = np.array([0.55, 0.55, 0.55], dtype=float)
PI1_HETERO = np.array([0.82, 0.68, 0.30], dtype=float)
PI1_HOMOGENEOUS = np.array([0.70, 0.70, 0.70], dtype=float)
PARAMETER_SEED = 20260821

_rng_param = np.random.default_rng(PARAMETER_SEED)
BETA = _rng_param.normal(0.0, 0.35, size=(K, D))
BIAS = _rng_param.normal(0.0, 0.12, size=K)
GROUP_EFFECT = _rng_param.normal(0.0, 0.15, size=(G, K))
NONLINEAR_ACTION_COEFF = _rng_param.normal(0.0, 0.30, size=K)

GROUP_MEANS = np.zeros((G, D), dtype=float)
GROUP_MEANS[0, 0] = 0.75
GROUP_MEANS[1, 0] = -0.75
GROUP_MEANS[2, 1] = 0.75
SHIFT_DIRECTION = np.ones(D, dtype=float) / np.sqrt(D)


SCENARIOS = {
    "S0_clean": {
        "base": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3,
                     tau=1.20, mismatch=0.00, exploration_floor=0.15,
                     deploy_shift=0.0, nonlinear_strength=0.0, target_variant="hetero")
    },
    "S1_overlap": {
        "good": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3,
                     tau=1.20, mismatch=0.10, exploration_floor=0.10,
                     deploy_shift=0.0, nonlinear_strength=0.0, target_variant="hetero"),
        "moderate": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3,
                         tau=0.70, mismatch=0.30, exploration_floor=0.05,
                         deploy_shift=0.0, nonlinear_strength=0.0, target_variant="hetero"),
        "poor": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3,
                     tau=0.35, mismatch=0.60, exploration_floor=0.02,
                     deploy_shift=0.0, nonlinear_strength=0.0, target_variant="hetero"),
        "severe": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3,
                       tau=0.15, mismatch=0.85, exploration_floor=0.005,
                       deploy_shift=0.0, nonlinear_strength=0.0, target_variant="hetero"),
    },
    "S2_rare_heterogeneous": {
        "balanced": dict(log_group_weights=[.34,.33,.33], deploy_group_weights=[.34,.33,.33], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
        "mild": dict(log_group_weights=[.55,.30,.15], deploy_group_weights=[.55,.30,.15], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
        "rare": dict(log_group_weights=[.70,.25,.05], deploy_group_weights=[.70,.25,.05], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
        "severe": dict(log_group_weights=[.89,.10,.01], deploy_group_weights=[.89,.10,.01], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
    },
    "S2_rare_homogeneous": {
        "balanced": dict(log_group_weights=[.34,.33,.33], deploy_group_weights=[.34,.33,.33], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="homogeneous"),
        "mild": dict(log_group_weights=[.55,.30,.15], deploy_group_weights=[.55,.30,.15], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="homogeneous"),
        "rare": dict(log_group_weights=[.70,.25,.05], deploy_group_weights=[.70,.25,.05], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="homogeneous"),
        "severe": dict(log_group_weights=[.89,.10,.01], deploy_group_weights=[.89,.10,.01], tau=1.20, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="homogeneous"),
    },
    "S3_shift": {
        "mean_0": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0.0, nonlinear_strength=0, target_variant="hetero"),
        "mean_025": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=.25, nonlinear_strength=0, target_variant="hetero"),
        "mean_05": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=.50, nonlinear_strength=0, target_variant="hetero"),
        "mean_10": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=1.00, nonlinear_strength=0, target_variant="hetero"),
        "composition_mild": dict(log_group_weights=[1/3]*3, deploy_group_weights=[.33,.33,.34], tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
        "composition_medium": dict(log_group_weights=[1/3]*3, deploy_group_weights=[.25,.25,.50], tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
        "composition_severe": dict(log_group_weights=[1/3]*3, deploy_group_weights=[.10,.10,.80], tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0, target_variant="hetero"),
    },
    "S4_misspecified": {
        "linear": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=0.0, target_variant="hetero"),
        "mild_nonlinear": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=.50, target_variant="hetero"),
        "strong_nonlinear": dict(log_group_weights=[1/3]*3, deploy_group_weights=[1/3]*3, tau=1.2, mismatch=0, exploration_floor=.15, deploy_shift=0, nonlinear_strength=1.0, target_variant="hetero"),
    },
    "S5_combined": {
        "moderate": dict(log_group_weights=[.70,.25,.05], deploy_group_weights=[.60,.20,.20], tau=.50, mismatch=.45, exploration_floor=.03, deploy_shift=.50, nonlinear_strength=.50, target_variant="hetero"),
        "severe": dict(log_group_weights=[.89,.10,.01], deploy_group_weights=[.15,.10,.75], tau=.20, mismatch=.75, exploration_floor=.01, deploy_shift=1.00, nonlinear_strength=1.00, target_variant="hetero"),
    },
}

S6_LEVELS = {
    "stationary": dict(kind="stationary", strength=0.0, context_drift=0.0),
    "gradual_recent_advantage": dict(kind="recent_advantage", strength=1.0, context_drift=0.25),
    "gradual_old_optimistic": dict(kind="old_optimistic", strength=1.0, context_drift=0.25),
    "abrupt_recent_advantage": dict(kind="abrupt_recent_advantage", strength=1.25, context_drift=0.35),
    "group_specific": dict(kind="group_specific", strength=1.25, context_drift=0.20),
}


def sigmoid(z):
    return expit(np.clip(z, -30, 30))


def softmax(z):
    z = np.asarray(z, float)
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_categorical(prob: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(prob, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((len(prob), 1))
    return (u > cdf).sum(axis=1).astype(int)


def sample_contexts(n, group_weights, shift_norm, rng, time=None, context_drift=0.0):
    wg = np.asarray(group_weights, float)
    wg /= wg.sum()
    g = rng.choice(G, size=n, p=wg)
    X = rng.normal(0, 1, size=(n, D)) + GROUP_MEANS[g]
    X += float(shift_norm) * SHIFT_DIRECTION
    if time is not None and context_drift != 0:
        # Older contexts differ modestly from the deployment-time distribution.
        age = 1.0 - np.asarray(time, float)
        X += float(context_drift) * age[:, None] * SHIFT_DIRECTION[None, :]
    return X, g.astype(int)


def base_logits(X, group, nonlinear_strength):
    logits = X @ BETA.T + BIAS[None, :] + GROUP_EFFECT[group]
    if nonlinear_strength != 0:
        f = X[:, 0] * X[:, 1] + 0.5 * np.sin(X[:, 2])
        logits += float(nonlinear_strength) * f[:, None] * NONLINEAR_ACTION_COEFF[None, :]
    return logits


def q_current(X, group, nonlinear_strength):
    return sigmoid(base_logits(X, group, nonlinear_strength))


def q_historical(X, group, time, nonlinear_strength, drift_kind, drift_strength):
    logits = base_logits(X, group, nonlinear_strength)
    if drift_kind == "stationary" or drift_strength == 0:
        return sigmoid(logits)
    t = np.asarray(time, float)
    age = 1.0 - t
    current_best = np.argmax(logits, axis=1)
    row = np.arange(len(X))
    adj = np.zeros_like(logits)
    if drift_kind == "recent_advantage":
        adj[row, current_best] -= drift_strength * age
    elif drift_kind == "old_optimistic":
        adj[row, current_best] += drift_strength * age
    elif drift_kind == "abrupt_recent_advantage":
        old = (t < 0.65).astype(float)
        adj[row, current_best] -= drift_strength * old
    elif drift_kind == "group_specific":
        # Stress group changes most; other groups stay nearly stationary.
        old = age * (group == 2)
        adj[row, current_best] += drift_strength * old
    else:
        raise ValueError(drift_kind)
    # Small opposite redistribution avoids making one coefficient the only change.
    adj -= adj.mean(axis=1, keepdims=True)
    return sigmoid(logits + adj)


def target_policies(q: np.ndarray, group: np.ndarray, variant: str):
    best = np.argmax(q, axis=1)
    one = np.zeros_like(q)
    one[np.arange(len(q)), best] = 1.0
    uni = np.full_like(q, 1.0 / K)
    w0 = PI0_ORACLE_WEIGHT[group][:, None]
    w1v = PI1_HETERO if variant == "hetero" else PI1_HOMOGENEOUS
    w1 = w1v[group][:, None]
    return w0 * one + (1-w0) * uni, w1 * one + (1-w1) * uni


def behavior_policy(q: np.ndarray, tau: float, mismatch: float, exploration_floor: float):
    rolled = np.roll(q, 1, axis=1)
    score = (1-mismatch) * q + mismatch * rolled
    mu = softmax(score / max(float(tau), 1e-8))
    mu = (1-exploration_floor) * mu + exploration_floor / K
    return mu / mu.sum(axis=1, keepdims=True)


def truth_table(X, group, q, pi0, pi1):
    v0 = np.sum(pi0 * q, axis=1)
    v1 = np.sum(pi1 * q, axis=1)
    delta = v1 - v0
    rows = []
    for g in range(G):
        m = group == g
        rows.append(dict(group=g, n_truth=int(m.sum()), population_weight=float(m.mean()),
                         v0_true=float(v0[m].mean()), v1_true=float(v1[m].mean()),
                         delta_true=float(delta[m].mean()),
                         fraction_pointwise_delta_negative=float((delta[m] < 0).mean())))
    rows.append(dict(group=-1, n_truth=len(group), population_weight=1.0,
                     v0_true=float(v0.mean()), v1_true=float(v1.mean()),
                     delta_true=float(delta.mean()),
                     fraction_pointwise_delta_negative=float((delta < 0).mean())))
    return pd.DataFrame(rows)


def log_mixture_density(X, group_weights, shift_norm):
    """Known mixture N(mean_g + shift, I) density, used only for synthetic oracle shift."""
    weights = np.asarray(group_weights, float)
    weights /= weights.sum()
    means = GROUP_MEANS + float(shift_norm) * SHIFT_DIRECTION[None, :]
    const = -0.5 * D * math.log(2 * math.pi)
    terms = []
    for g in range(G):
        d2 = np.sum((X - means[g]) ** 2, axis=1)
        terms.append(math.log(max(weights[g], 1e-15)) + const - 0.5 * d2)
    return logsumexp(np.vstack(terms), axis=0)


def oracle_density_ratio(X, log_w, dep_w, dep_shift):
    lr = log_mixture_density(X, dep_w, dep_shift) - log_mixture_density(X, log_w, 0.0)
    return np.clip(np.exp(np.clip(lr, -8, 8)), 0.05, 20.0)


def standard_truth_reference(c, seed):
    """High-precision truth reference, frozen once per scenario/level.

    The estimand is a property of the DGP, not of each logging replication.  The old
    runner regenerated 100k truth rows for every replication, adding large compute
    cost and tiny Monte Carlo jitter.  Freezing one independent 100k reference per
    condition preserves (and slightly improves) reference precision.
    """
    rngt = np.random.default_rng(seed)
    Xt, gt = sample_contexts(cfg.SYN_TRUTH_SIZE, c["deploy_group_weights"], c["deploy_shift"], rngt)
    qt = q_current(Xt, gt, c["nonlinear_strength"])
    p0t, p1t = target_policies(qt, gt, c["target_variant"])
    return truth_table(Xt, gt, qt, p0t, p1t)


def generate_standard_condition(scenario, level, c, n, seed, truth_reference=None):
    rng = np.random.default_rng(seed)
    X, g = sample_contexts(n, c["log_group_weights"], 0.0, rng)
    q = q_current(X, g, c["nonlinear_strength"])
    pi0, pi1 = target_policies(q, g, c["target_variant"])
    mu = behavior_policy(q, c["tau"], c["mismatch"], c["exploration_floor"])
    a = sample_categorical(mu, rng)
    p = mu[np.arange(n), a]
    r = rng.binomial(1, q[np.arange(n), a]).astype(float)
    data = BanditData(X=X, action=a, reward=r, pscore=p, pi0=pi0, pi1=pi1, group=g,
                      sample_id=np.arange(n))
    validate_bandit_data(data, f"{scenario}/{level}")

    if truth_reference is None:
        truth_reference = standard_truth_reference(c, seed + 100_000)
    truth = truth_reference.copy()

    # Deployment sample remains replication-specific because it is observed by the
    # estimated density-ratio procedure in S3 and therefore belongs to estimator noise.
    rngd = np.random.default_rng(seed + 200_000)
    Xd, gd = sample_contexts(cfg.SYN_DEPLOY_SIZE, c["deploy_group_weights"], c["deploy_shift"], rngd)
    return data, truth, Xd


def s6_truth_reference(seed):
    """Freeze the common current-time truth used by all S6 levels."""
    rngt = np.random.default_rng(seed)
    Xt, gt = sample_contexts(cfg.SYN_TRUTH_SIZE, [1/3]*3, 0.0, rngt)
    qt = q_current(Xt, gt, 0.0)
    p0t, p1t = target_policies(qt, gt, "hetero")
    return truth_table(Xt, gt, qt, p0t, p1t)


def generate_s6_condition(level, c, n, seed, truth_reference=None):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, n)
    X, g = sample_contexts(n, [1/3]*3, 0.0, rng, time=time, context_drift=c["context_drift"])
    qT_logx = q_current(X, g, 0.0)
    qhist = q_historical(X, g, time, 0.0, c["kind"], c["strength"])
    pi0, pi1 = target_policies(qT_logx, g, "hetero")
    mu = behavior_policy(qhist, 1.0, 0.10, 0.10)
    a = sample_categorical(mu, rng)
    p = mu[np.arange(n), a]
    r = rng.binomial(1, qhist[np.arange(n), a]).astype(float)
    data = BanditData(X=X, action=a, reward=r, pscore=p, pi0=pi0, pi1=pi1, group=g,
                      sample_id=np.arange(n), time=time)
    validate_bandit_data(data, f"S6/{level}")

    if truth_reference is None:
        truth_reference = s6_truth_reference(seed + 100_000)
    truth = truth_reference.copy()
    rngd = np.random.default_rng(seed + 200_000)
    Xd, gd = sample_contexts(cfg.SYN_DEPLOY_SIZE, [1/3]*3, 0.0, rngd)
    return data, truth, Xd


def _pooling_extension(ev, truth, meta, seed):
    """S2 independent vs complete vs empirical-Bayes partial pooling."""
    ind = {g: ev.draws["bb_dr"][g] for g in range(G)}
    est = {g: float(np.mean(ind[g])) for g in range(G)}
    ses = {g: float(np.std(ind[g], ddof=1)) for g in range(G)}
    hier = empirical_bayes_partial_pooling(est, ses, cfg.BAYES_BOOTSTRAP_DRAWS, seed + 701)
    complete = complete_pooling_draws(ev.components.dr_delta, ev.components.dr_delta.astype(int)*0 + np.asarray([0]*len(ev.components.dr_delta)), 1, seed) if False else None
    # Correct complete pooling: global BB draw copied to all true groups.
    global_draw = ev.draws["bb_dr"][-1]
    complete = {g: global_draw.copy() for g in range(G)}
    rows_g, rows_d, rows_f, rows_c = [], [], [], []
    for name, dd in [("bb_dr_independent", ind), ("bb_dr_complete_pool", complete), ("hier_eb_dr", hier)]:
        s = summaries_from_draw_dict(dd, cfg.ALPHA, cfg.DELTA)
        s["method"] = name
        s = s.merge(truth[["group", "delta_true"]], on="group", how="left")
        for k, v in meta.items(): s[k] = v
        rows_g.append(s)
        rows_d.append({**meta, "method": name, **decision_metrics_from_truth(s[["group","gate"]], truth)})
        fr = safety_coverage_frontier_from_draws(dd, truth)
        fr["method"] = name
        for k, v in meta.items(): fr[k] = v
        rows_f.append(fr)
        for nominal in cfg.COVERAGE_LEVELS:
            tail=(1-nominal)/2
            for g in range(G):
                lo,hi=np.quantile(dd[g],[tail,1-tail])
                tv=float(truth.loc[truth.group==g,"delta_true"].iloc[0])
                rows_c.append({**meta,"method":name,"group":g,"nominal":nominal,"covered":int(lo<=tv<=hi),"interval_lower":lo,"interval_upper":hi,"interval_width":hi-lo})
    return pd.concat(rows_g,ignore_index=True), pd.DataFrame(rows_d), pd.concat(rows_f,ignore_index=True), pd.DataFrame(rows_c)


def _shift_extension(ev, data, truth, Xdeploy, c, meta, seed):
    """Estimated/oracle density-ratio weighted BB-DR variants for S3."""
    r_est = estimate_density_ratio_log_to_deploy(data.X, Xdeploy, seed + 801)
    r_oracle = oracle_density_ratio(data.X, c["log_group_weights"], c["deploy_group_weights"], c["deploy_shift"])
    rows_g, rows_d, rows_f = [], [], []
    for name, rr in [("bb_dr_shift_estimated",r_est),("bb_dr_shift_oracle",r_oracle)]:
        dd=dr_uncertainty_by_group(ev.components.dr_delta,data.group,cfg.BAYES_BOOTSTRAP_DRAWS,seed+811,base_weight=rr,bayesian=True)
        s=summaries_from_draw_dict(dd,cfg.ALPHA,cfg.DELTA)
        s["method"]=name
        s=s.merge(truth[["group","delta_true"]],on="group",how="left")
        diag=evidence_diagnostics(data,ev.components,base_weight=rr)
        s=s.merge(diag,on="group",how="left")
        for k,v in meta.items(): s[k]=v
        rows_g.append(s)
        rows_d.append({**meta,"method":name,**decision_metrics_from_truth(s[["group","gate"]],truth)})
        fr=safety_coverage_frontier_from_draws(dd,truth);fr["method"]=name
        for k,v in meta.items():fr[k]=v
        rows_f.append(fr)
    # Ratio diagnostics saved as group=-999 pseudo rows in gate table for auditability.
    return pd.concat(rows_g,ignore_index=True), pd.DataFrame(rows_d), pd.concat(rows_f,ignore_index=True), pd.DataFrame({
        **{k:[v,v] for k,v in meta.items()},
        "ratio_type":["estimated","oracle"],
        "ratio_mean":[float(np.mean(r_est)),float(np.mean(r_oracle))],
        "ratio_q99":[float(np.quantile(r_est,.99)),float(np.quantile(r_oracle,.99))],
        "ratio_max":[float(np.max(r_est)),float(np.max(r_oracle))],
    })


def _plot_synthetic_outputs(point, gates, decisions, calibration, frontiers, diagnostics):
    out = cfg.FIGURES_DIR / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    # Figure: global-good / local-harm truth structure.
    d = gates[(gates.scenario=="S0_clean") & (gates.method=="bb_dr") & (gates.group>=0)]
    if len(d):
        t=d.groupby("group",as_index=False)["delta_true"].mean()
        fig,ax=plt.subplots(figsize=(6.5,4.8)); ax.bar(t.group.astype(str),t.delta_true)
        ax.axhline(0,linewidth=1);ax.set_xlabel("Subgroup");ax.set_ylabel("True local improvement")
        ax.set_title("Synthetic S0: global-good can contain local harm")
        save_show(fig,out/"fig_syn_truth_structure.png")

    # Calibration.
    c=calibration[(calibration.scenario=="S0_clean") & (calibration.group>=0)]
    if len(c):
        cc=c.groupby(["method","nominal"],as_index=False).covered.mean().rename(columns={"covered":"empirical"})
        plot_calibration(cc,"Synthetic S0 interval calibration",out/"fig_syn_s0_calibration.png")

    # S1 ESS vs width/error.
    s=gates[(gates.scenario=="S1_overlap") & (gates.method=="bb_dr") & (gates.group>=0)]
    if len(s):
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter(s.ess_fraction,s.interval_width,alpha=.6)
        ax.set_xlabel("Realized ESS / n");ax.set_ylabel("95% interval width");ax.set_title("S1: overlap evidence vs uncertainty")
        save_show(fig,out/"fig_syn_s1_ess_vs_width.png")

    # S2 pooling false deployment/coverage summary.
    s=decisions[decisions.scenario.str.startswith("S2_",na=False)]
    if len(s):
        ss=s.groupby(["scenario","level","method"],as_index=False)[["harmful_exposure","deployment_coverage"]].mean()
        save_csv(ss,cfg.TABLES_DIR/"synthetic_s2_pooling_summary.csv")

    # S3 shift.
    s=decisions[decisions.scenario=="S3_shift"]
    if len(s):
        fig,ax=plt.subplots(figsize=(8,5))
        for m,dd in s.groupby("method"):
            tmp=dd.groupby("level",as_index=False).harmful_exposure.mean()
            ax.plot(np.arange(len(tmp)),tmp.harmful_exposure,marker="o",label=m)
        ax.set_xticks(np.arange(len(tmp)));ax.set_xticklabels(tmp.level,rotation=45,ha="right")
        ax.set_ylabel("Harmful exposure");ax.set_title("S3: shift and deployment harm");ax.legend(fontsize=7)
        save_show(fig,out/"fig_syn_s3_shift_harm.png")

    # S4 misspecification bias/RMSE.
    s=point[(point.scenario=="S4_misspecified") & (point.group>=0)]
    if len(s):
        ss=s.groupby(["level","estimator"],as_index=False).sq_error.mean();ss["rmse"]=np.sqrt(ss.sq_error)
        fig,ax=plt.subplots(figsize=(7.5,5))
        for m,dd in ss.groupby("estimator"): ax.plot(dd.level,dd.rmse,marker="o",label=m)
        ax.set_ylabel("RMSE");ax.set_title("S4: reward-model misspecification");ax.tick_params(axis='x',rotation=30);ax.legend()
        save_show(fig,out/"fig_syn_s4_rmse.png")

    # S5 frontier averaged over replications at each threshold.
    f=frontiers[frontiers.scenario=="S5_combined"]
    if len(f):
        ff=f.groupby(["method","posterior_threshold"],as_index=False)[["deployment_coverage","harmful_exposure"]].mean()
        plot_safety_coverage(ff,"S5 combined-stress Safety-Coverage Frontier",out/"fig_syn_s5_frontier.png")

    # S6 historical memory metrics.
    s=gates[(gates.scenario=="S6_temporal") & (gates.method=="bb_dr") & (gates.group>=0)]
    if len(s):
        ss=s.groupby(["level","relevance_scheme"],as_index=False).agg(
            abs_error=("delta_true",lambda x: np.nan),
            interval_width=("interval_width","mean"),weighted_ess=("weighted_ess","mean"))
        # Calculate error from draw mean explicitly.
        tmp=s.copy();tmp["abs_error"]=(tmp.estimate_draw_mean-tmp.delta_true).abs()
        ss=tmp.groupby(["level","relevance_scheme"],as_index=False).agg(abs_error=("abs_error","mean"),interval_width=("interval_width","mean"),weighted_ess=("weighted_ess","mean"))
        save_csv(ss,cfg.TABLES_DIR/"synthetic_s6_memory_summary.csv")
        fig,ax=plt.subplots(figsize=(9,5))
        for lev,dd in ss.groupby("level"): ax.plot(dd.relevance_scheme,dd.abs_error,marker="o",label=lev)
        ax.set_ylabel("Mean absolute local-improvement error");ax.set_title("S6: historical relevance vs estimation error");ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=7)
        save_show(fig,out/"fig_syn_s6_memory_error.png")


RUNNER_REVISION = "PAPER-EFFICIENT-SYN-R2"
KEY_CONDITIONS = {
    ("S0_clean", "base"),
    ("S1_overlap", "severe"),
    ("S2_rare_heterogeneous", "severe"),
    ("S5_combined", "severe"),
}


def _synthetic_output_paths():
    names = [
        "synthetic_point_estimates_long.csv",
        "synthetic_gate_results_long.csv",
        "synthetic_decision_results.csv",
        "synthetic_calibration_long.csv",
        "synthetic_safety_coverage_frontiers.csv",
        "synthetic_evidence_diagnostics.csv",
        "synthetic_shift_ratio_diagnostics.csv",
        "synthetic_calibration_summary.csv",
        "synthetic_decision_summary.csv",
        "synthetic_point_summary.csv",
        "synthetic_s2_pooling_summary.csv",
        "synthetic_s6_memory_summary.csv",
    ]
    return [cfg.TABLES_DIR / x for x in names] + [cfg.RESULTS_DIR / "synthetic_run_manifest.json"]


def _clear_previous_synthetic_outputs():
    """Clear only final Synthetic artifacts; never touch Semi-Synthetic or OBD outputs."""
    for path in _synthetic_output_paths():
        if path.exists():
            path.unlink()
    fig_dir = cfg.FIGURES_DIR / "synthetic"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)


def _protocol_payload(conditions, s6_items, schemes, quick):
    return {
        "runner_revision": RUNNER_REVISION,
        "run_mode": cfg.RUN_MODE,
        "quick": bool(quick),
        "seed": int(cfg.RANDOM_SEED),
        "n_folds": int(cfg.N_FOLDS),
        "bootstrap_draws": int(cfg.BOOTSTRAP_DRAWS),
        "bayes_bootstrap_draws": int(cfg.BAYES_BOOTSTRAP_DRAWS),
        "sample_sizes": list(map(int, cfg.SYN_SAMPLE_SIZES)),
        "reps": int(cfg.SYN_REPS),
        "key_reps": int(cfg.SYN_KEY_REPS),
        "truth_size": int(cfg.SYN_TRUTH_SIZE),
        "deploy_size": int(cfg.SYN_DEPLOY_SIZE),
        "checkpoint_every": int(getattr(cfg, "SYN_CHECKPOINT_EVERY", 20)),
        "s6_sample_sizes": list(map(int, getattr(cfg, "SYN_S6_SAMPLE_SIZES", cfg.SYN_SAMPLE_SIZES))),
        "s6_reps": int(getattr(cfg, "SYN_S6_REPS", cfg.SYN_REPS)),
        "s0_s5_conditions": [[a, b] for a, b, _ in conditions],
        "s6_levels": [a for a, _ in s6_items],
        "s6_schemes": [x["name"] for x in schemes],
    }


def _protocol_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _prepare_checkpoint_dir(payload):
    root = cfg.CACHE_DIR / "synthetic_paper_checkpoint_efficient_v2"
    sig_path = root / "protocol.json"
    sig = _protocol_hash(payload)
    force = os.environ.get("BCSPI_SYNTHETIC_FORCE_RESTART", "0") == "1"

    if force and root.exists():
        shutil.rmtree(root)

    if root.exists() and sig_path.exists():
        try:
            old = json.loads(sig_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if old.get("protocol_hash") != sig:
            print("[CHECKPOINT] Protocol changed; clearing incompatible Synthetic checkpoint cache.")
            shutil.rmtree(root)
    elif root.exists() and any(root.iterdir()):
        print("[CHECKPOINT] Unidentified Synthetic checkpoint cache; clearing for safety.")
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)
    record = {**payload, "protocol_hash": sig}
    sig_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return root, sig


def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_")


def _save_shard(path: Path, payload: dict):
    """Atomic checkpoint write; a crash cannot leave a half-valid .pkl shard."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_shards(root: Path):
    buckets = {k: [] for k in ["point", "gates", "decisions", "calibration", "frontiers", "diagnostics", "ratios"]}
    shards = sorted(root.glob("*.pkl"))
    if not shards:
        raise RuntimeError("No Synthetic checkpoint shards were produced.")
    for path in shards:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        for k in buckets:
            df = obj.get(k)
            if isinstance(df, pd.DataFrame) and not df.empty:
                buckets[k].append(df)
    out = {}
    for k, frames in buckets.items():
        out[k] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, shards


def _payload_from_lists(point_all, gate_all, decision_all, calib_all, frontier_all, diag_all, ratio_all):
    def cat(xs):
        return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return {
        "point": cat(point_all),
        "gates": cat(gate_all),
        "decisions": cat(decision_all),
        "calibration": cat(calib_all),
        "frontiers": cat(frontier_all),
        "diagnostics": cat(diag_all),
        "ratios": cat(ratio_all),
    }


def run_synthetic():
    cfg.print_protocol()
    print("\n[RUN] Controlled Synthetic S0--S6")
    print(f"Runner revision : {RUNNER_REVISION}")
    print("Formal compute-efficient Paper protocol: n is capped at 10,000; n=20,000 is not run.")
    print("Checkpoint shards are saved during the run and reused automatically after interruption.\n")

    quick = os.environ.get("BCSPI_INTERNAL_QUICK_TEST", "0") == "1"
    if not quick and cfg.RUN_MODE != "paper":
        print("[WARNING] RUN_MODE is not 'paper'; this will produce a smoke-scale result.")

    conditions = []
    for sc, levels in SCENARIOS.items():
        for lev, c in levels.items():
            conditions.append((sc, lev, c))
    if quick:
        conditions = [
            ("S0_clean", "base", SCENARIOS["S0_clean"]["base"]),
            ("S1_overlap", "severe", SCENARIOS["S1_overlap"]["severe"]),
            ("S4_misspecified", "strong_nonlinear", SCENARIOS["S4_misspecified"]["strong_nonlinear"]),
        ]

    s6_items = list(S6_LEVELS.items())
    allowed_schemes = set(getattr(cfg, "SYN_S6_SCHEME_NAMES", [x["name"] for x in cfg.RECENCY_SCHEMES]))
    schemes = [x for x in cfg.RECENCY_SCHEMES if x["name"] in allowed_schemes]
    if cfg.RUN_MODE == "smoke":
        keep = {"full", "roll_50", "decay_h25", "relevance_h25"}
        schemes = [x for x in schemes if x["name"] in keep]
    if quick:
        s6_items = [("gradual_recent_advantage", S6_LEVELS["gradual_recent_advantage"])]
        schemes = [x for x in cfg.RECENCY_SCHEMES if x["name"] in {"full", "decay_h25"}]

    payload = _protocol_payload(conditions, s6_items, schemes, quick)
    checkpoint_dir, protocol_hash = _prepare_checkpoint_dir(payload)

    # Remove stale smoke/old-paper Synthetic FINAL outputs immediately.  Completed work
    # lives safely in checkpoint shards until the new final tables are assembled.
    _clear_previous_synthetic_outputs()

    batch_size = 1 if quick else max(1, int(getattr(cfg, "SYN_CHECKPOINT_EVERY", 20)))

    # Freeze high-precision truth once per standard condition.  This is independent of
    # behavior-log replications and avoids regenerating 100k truth rows hundreds of times.
    truth_cache = {}
    for ci, (scenario, level, c) in enumerate(conditions):
        truth_seed = cfg.RANDOM_SEED + 50_000_000 + ci
        truth_cache[(scenario, level)] = standard_truth_reference(c, truth_seed)

    # ----------------------- S0--S5 -----------------------
    for n in cfg.SYN_SAMPLE_SIZES:
        for ci, (scenario, level, c) in enumerate(conditions):
            reps = cfg.SYN_KEY_REPS if (cfg.RUN_MODE == "paper" and (scenario, level) in KEY_CONDITIONS) else cfg.SYN_REPS
            if quick:
                reps = 1
            truth_ref = truth_cache[(scenario, level)]

            for start in range(0, reps, batch_size):
                stop = min(start + batch_size, reps)
                shard = checkpoint_dir / (
                    f"std__n{int(n):05d}__{_safe_name(scenario)}__{_safe_name(level)}__r{start:04d}-{stop-1:04d}.pkl"
                )
                if shard.exists():
                    print(f"[RESUME] {shard.name}")
                    continue

                point_all = []; gate_all = []; decision_all = []; calib_all = []; frontier_all = []; diag_all = []; ratio_all = []
                for rep in range(start, stop):
                    seed = cfg.RANDOM_SEED + n + 100_000 * ci + rep
                    print(f"[Synthetic] n={n} {scenario}/{level} rep={rep+1}/{reps}")
                    data, truth, Xd = generate_standard_condition(
                        scenario, level, c, n, seed, truth_reference=truth_ref
                    )
                    models = ["linear"]
                    if scenario == "S4_misspecified" and cfg.RUN_MODE == "paper":
                        models = ["linear", "flexible"]
                    for model in models:
                        meta = {"layer": "synthetic", "scenario": scenario, "level": level, "rep": rep, "seed": seed, "n": n}
                        ev = evaluate_generic_bandit(data, truth, meta, model, seed)
                        point_all.append(ev.point); gate_all.append(ev.gates); decision_all.append(ev.decisions)
                        calib_all.append(ev.calibration); frontier_all.append(ev.frontiers); diag_all.append(ev.diagnostics)
                        if scenario.startswith("S2_") and model == "linear":
                            pg, pdec, pf, pc = _pooling_extension(ev, truth, {**meta, "nuisance_model": model}, seed)
                            gate_all.append(pg); decision_all.append(pdec); frontier_all.append(pf); calib_all.append(pc)
                        if scenario == "S3_shift" and model == "linear":
                            sg, sd, sf, sr = _shift_extension(ev, data, truth, Xd, c, {**meta, "nuisance_model": model}, seed)
                            gate_all.append(sg); decision_all.append(sd); frontier_all.append(sf); ratio_all.append(sr)

                _save_shard(shard, _payload_from_lists(
                    point_all, gate_all, decision_all, calib_all, frontier_all, diag_all, ratio_all
                ))
                print(f"[CHECKPOINT] saved {shard.name}")

    # ----------------------- S6 temporal robustness -----------------------
    s6_truth = s6_truth_reference(cfg.RANDOM_SEED + 60_000_000)
    s6_sample_sizes = cfg.SYN_S6_SAMPLE_SIZES if not quick else [cfg.SYN_SAMPLE_SIZES[0]]
    s6_reps = cfg.SYN_S6_REPS if not quick else 1

    for n in s6_sample_sizes:
        for li, (level, c) in enumerate(s6_items):
            for start in range(0, s6_reps, batch_size):
                stop = min(start + batch_size, s6_reps)
                shard = checkpoint_dir / (
                    f"s6__n{int(n):05d}__{_safe_name(level)}__r{start:04d}-{stop-1:04d}.pkl"
                )
                if shard.exists():
                    print(f"[RESUME] {shard.name}")
                    continue

                point_all = []; gate_all = []; decision_all = []; calib_all = []; frontier_all = []; diag_all = []
                for rep in range(start, stop):
                    seed = cfg.RANDOM_SEED + 9_000_000 + n + li * 100_000 + rep
                    data, truth, Xd = generate_s6_condition(level, c, n, seed, truth_reference=s6_truth)
                    for scheme in schemes:
                        print(f"[S6] n={n} {level} {scheme['name']} rep={rep+1}/{s6_reps}")
                        r = recency_weights(data.time, scheme, X_log=data.X, X_deploy=Xd, seed=seed + 3)
                        age = (np.max(data.time) - data.time) / max(np.ptp(data.time), 1e-12)
                        meta = {
                            "layer": "synthetic", "scenario": "S6_temporal", "level": level,
                            "rep": rep, "seed": seed, "n": n,
                            "relevance_scheme": scheme["name"],
                            "window": scheme["param"] if scheme["kind"] == "rolling" else np.nan,
                            "half_life": scheme["param"] if scheme["kind"] in {"exp", "relevance"} else np.nan,
                            "drift_strength": c["strength"], "current_time": 1.0,
                            "mean_obs_age": float(np.mean(age)),
                        }
                        ev = evaluate_generic_bandit(data, truth, meta, "linear", seed, base_weight=r)
                        point_all.append(ev.point); gate_all.append(ev.gates); decision_all.append(ev.decisions)
                        calib_all.append(ev.calibration); frontier_all.append(ev.frontiers); diag_all.append(ev.diagnostics)

                _save_shard(shard, _payload_from_lists(
                    point_all, gate_all, decision_all, calib_all, frontier_all, diag_all, []
                ))
                print(f"[CHECKPOINT] saved {shard.name}")

    # ----------------------- Assemble final paper outputs -----------------------
    all_data, shard_paths = _load_shards(checkpoint_dir)
    point = all_data["point"]
    gates = all_data["gates"]
    decisions = all_data["decisions"]
    calibration = all_data["calibration"]
    frontiers = all_data["frontiers"]
    diagnostics = all_data["diagnostics"]
    ratios = all_data["ratios"]

    required = {
        "point": point, "gates": gates, "decisions": decisions,
        "calibration": calibration, "frontiers": frontiers, "diagnostics": diagnostics,
    }
    empty = [k for k, v in required.items() if v.empty]
    if empty:
        raise RuntimeError(f"Synthetic final assembly failed; empty result tables: {empty}")

    for df in [point, gates, decisions, calibration, frontiers, diagnostics]:
        for col in ["relevance_scheme", "window", "half_life", "drift_strength", "current_time", "mean_obs_age"]:
            if col not in df:
                df[col] = np.nan

    base = cfg.TABLES_DIR
    save_csv(point, base / "synthetic_point_estimates_long.csv")
    save_csv(gates, base / "synthetic_gate_results_long.csv")
    save_csv(decisions, base / "synthetic_decision_results.csv")
    save_csv(calibration, base / "synthetic_calibration_long.csv")
    save_csv(frontiers, base / "synthetic_safety_coverage_frontiers.csv")
    save_csv(diagnostics, base / "synthetic_evidence_diagnostics.csv")
    if len(ratios):
        save_csv(ratios, base / "synthetic_shift_ratio_diagnostics.csv")

    calib_summary = calibration.groupby(
        ["scenario", "level", "method", "nominal"], dropna=False, as_index=False
    ).covered.mean().rename(columns={"covered": "empirical"})
    dec_summary = monte_carlo_summary(
        decisions, ["scenario", "level", "method"],
        ["deployment_coverage", "harmful_exposure", "selective_value", "oracle_regret"]
    )
    point_summary = monte_carlo_summary(
        point, ["scenario", "level", "estimator", "group"],
        ["error", "abs_error", "sq_error"]
    )
    save_csv(calib_summary, base / "synthetic_calibration_summary.csv")
    save_csv(dec_summary, base / "synthetic_decision_summary.csv")
    save_csv(point_summary, base / "synthetic_point_summary.csv")

    _plot_synthetic_outputs(point, gates, decisions, calibration, frontiers, diagnostics)

    print_frame("Synthetic S0 calibration summary", calib_summary[calib_summary.scenario == "S0_clean"], 30)
    print_frame("Synthetic decision summary (selected rows)", dec_summary, 40)
    s1 = gates[(gates.scenario == "S1_overlap") & (gates.method == "bb_dr") & (gates.group >= 0)]
    if len(s1):
        print("\n[S1 diagnostic] corr(ESS fraction, interval width) =", round(float(s1[["ess_fraction", "interval_width"]].corr().iloc[0, 1]), 4))
    s6 = gates[(gates.scenario == "S6_temporal") & (gates.method == "bb_dr") & (gates.group >= 0)].copy()
    if len(s6):
        s6["abs_error"] = (s6.estimate_draw_mean - s6.delta_true).abs()
        print_frame(
            "S6 historical-relevance summary",
            s6.groupby(["level", "relevance_scheme"], as_index=False).agg(
                abs_error=("abs_error", "mean"), interval_width=("interval_width", "mean"),
                weighted_ess=("weighted_ess", "mean")
            ), 50
        )

    # Final internal guard against accidentally producing a smoke-looking paper file.
    observed_n = sorted(pd.Series(point["n"]).dropna().astype(int).unique().tolist())
    if cfg.RUN_MODE == "paper" and not set(cfg.SYN_SAMPLE_SIZES).issubset(set(observed_n)):
        raise RuntimeError(
            f"Paper-mode Synthetic output is missing configured sample sizes. observed={observed_n}, expected core={cfg.SYN_SAMPLE_SIZES}"
        )

    manifest = {
        **payload,
        "protocol_hash": protocol_hash,
        "compute_efficient_paper_protocol": True,
        "n_20000_intentionally_omitted": True,
        "truth_reference_frozen_per_condition": True,
        "checkpoint_dir": str(checkpoint_dir),
        "completed_checkpoint_shards": len(shard_paths),
        "final_point_rows": int(len(point)),
        "final_gate_rows": int(len(gates)),
        "observed_n_values": observed_n,
    }
    save_json(manifest, cfg.RESULTS_DIR / "synthetic_run_manifest.json")

    print("\nSynthetic PAPER run completed without fatal validation errors.")
    print(f"Observed n values : {observed_n}")
    print(f"Protocol hash     : {protocol_hash}")
    print(f"Tables            : {cfg.TABLES_DIR}")
    print(f"Figures           : {cfg.FIGURES_DIR / 'synthetic'}")
    print(f"Checkpoint cache  : {checkpoint_dir}")
    return point, gates, decisions, calibration, frontiers, diagnostics


if __name__ == "__main__":
    run_synthetic()
