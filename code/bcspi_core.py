"""Shared BC-SPI data contract, OPE, uncertainty, Gate, metrics, and plotting helpers.

This module contains no dataset-specific truth logic.  Synthetic/semi truth and
OBD on-policy references are passed in only AFTER OPE fitting has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit, logsumexp
from scipy.stats import norm

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bcspi_config as cfg


EPS = 1e-12


# ============================================================================
# Unified data contract
# ============================================================================
@dataclass
class BanditData:
    X: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    pi0: np.ndarray
    pi1: np.ndarray
    group: np.ndarray
    sample_id: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    action_context: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return int(self.pi0.shape[1])


@dataclass
class OPEComponents:
    qhat: np.ndarray
    q_obs: np.ndarray
    dm0: np.ndarray
    dm1: np.ndarray
    ips0: np.ndarray
    ips1: np.ndarray
    dr0: np.ndarray
    dr1: np.ndarray
    switch0: np.ndarray
    switch1: np.ndarray
    w0: np.ndarray
    w1: np.ndarray

    @property
    def dm_delta(self) -> np.ndarray:
        return self.dm1 - self.dm0

    @property
    def ips_delta(self) -> np.ndarray:
        return self.ips1 - self.ips0

    @property
    def dr_delta(self) -> np.ndarray:
        return self.dr1 - self.dr0

    @property
    def switch_delta(self) -> np.ndarray:
        return self.switch1 - self.switch0


@dataclass
class OBDComponents:
    q_obs: np.ndarray
    q_pi0: np.ndarray
    q_pi1: np.ndarray
    dr0: np.ndarray
    dr1: np.ndarray
    w0: np.ndarray
    w1: np.ndarray

    @property
    def dr_delta(self) -> np.ndarray:
        return self.dr1 - self.dr0


# ============================================================================
# Validation / no-leakage checks
# ============================================================================
def validate_bandit_data(data: BanditData, name: str = "bandit") -> None:
    n = data.n
    arrays_1d = {
        "action": data.action,
        "reward": data.reward,
        "pscore": data.pscore,
        "group": data.group,
    }
    for key, arr in arrays_1d.items():
        if len(arr) != n:
            raise ValueError(f"{name}: {key} has len={len(arr)} but X has n={n}.")
    if data.pi0.shape != data.pi1.shape or data.pi0.shape[0] != n:
        raise ValueError(f"{name}: pi0/pi1 shapes do not align with X.")
    if np.any(~np.isfinite(data.X)) or np.any(~np.isfinite(data.reward)):
        raise ValueError(f"{name}: non-finite X or reward detected.")
    if np.any(data.pscore <= 0) or np.any(data.pscore > 1 + 1e-8):
        raise ValueError(f"{name}: pscore must be in (0,1].")
    if np.any(data.action < 0) or np.any(data.action >= data.k):
        raise ValueError(f"{name}: action outside [0,K).")
    for pname, p in [("pi0", data.pi0), ("pi1", data.pi1)]:
        if np.any(p < -1e-10):
            raise ValueError(f"{name}: {pname} contains negative probability.")
        err = np.max(np.abs(p.sum(axis=1) - 1.0))
        if err > 1e-6:
            raise ValueError(f"{name}: {pname} row sums deviate from 1 by {err:.3g}.")
    if data.position is not None and len(data.position) != n:
        raise ValueError(f"{name}: position length mismatch.")
    if data.time is not None and len(data.time) != n:
        raise ValueError(f"{name}: time length mismatch.")


def assert_truth_not_in_columns(columns: Iterable[str]) -> None:
    bad_tokens = ("truth", "y_true", "counterfactual", "oracle_reward", "q_true")
    bad = [c for c in columns if any(tok in c.lower() for tok in bad_tokens)]
    if bad:
        raise ValueError(f"Estimator-visible table appears to contain truth columns: {bad}")


# ============================================================================
# Generic synthetic/semi reward models
# ============================================================================
def _linear_action_features(X: np.ndarray, action: np.ndarray, k: int) -> np.ndarray:
    """Action-specific intercept and slope blocks: K * (1+d) columns."""
    X = np.asarray(X, dtype=float)
    action = np.asarray(action, dtype=int)
    n, d = X.shape
    out = np.zeros((n, k * (d + 1)), dtype=np.float32)
    row = np.arange(n)
    base = action * (d + 1)
    out[row, base] = 1.0
    for j in range(d):
        out[row, base + 1 + j] = X[:, j]
    return out


def _flex_action_features(X: np.ndarray, action: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    a_onehot = np.eye(k, dtype=np.float32)[np.asarray(action, dtype=int)]
    return np.hstack([X.astype(np.float32), a_onehot])


def _make_generic_model(model_kind: str, seed: int):
    if model_kind == "linear":
        return Ridge(alpha=1.0)
    if model_kind == "flexible":
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=150 if cfg.RUN_MODE == "paper" else 80,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            random_state=seed,
        )
    raise ValueError(f"Unknown model_kind={model_kind!r}")


def crossfit_qhat(
    data: BanditData,
    model_kind: str = "linear",
    n_folds: Optional[int] = None,
    seed: int = 0,
    fit_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Out-of-fold qhat(x,a) for all actions; used by Synthetic/Semi.

    fit_weight can encode S6 temporal relevance.  Final OPE aggregation weights
    are handled separately, so model weighting and estimator weighting remain explicit.
    """
    validate_bandit_data(data)
    n_folds = int(n_folds or cfg.N_FOLDS)
    n_folds = max(2, min(n_folds, data.n))
    fit_weight = np.ones(data.n, dtype=float) if fit_weight is None else np.asarray(fit_weight, dtype=float)
    if np.any(fit_weight < 0) or not np.isfinite(fit_weight).all():
        raise ValueError("fit_weight must be finite and nonnegative.")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    qhat = np.zeros((data.n, data.k), dtype=np.float32)

    for fold, (tr, te) in enumerate(kf.split(data.X)):
        model = _make_generic_model(model_kind, seed + fold + 1)
        if model_kind == "linear":
            Xtr = _linear_action_features(data.X[tr], data.action[tr], data.k)
        else:
            Xtr = _flex_action_features(data.X[tr], data.action[tr], data.k)

        try:
            model.fit(Xtr, data.reward[tr], sample_weight=fit_weight[tr])
        except TypeError:
            model.fit(Xtr, data.reward[tr])

        for a in range(data.k):
            aa = np.full(len(te), a, dtype=int)
            if model_kind == "linear":
                Xte = _linear_action_features(data.X[te], aa, data.k)
            else:
                Xte = _flex_action_features(data.X[te], aa, data.k)
            pred = np.asarray(model.predict(Xte), dtype=float)
            # Both current Synthetic and Semi rewards are bounded in [0,1].
            qhat[te, a] = np.clip(pred, 0.0, 1.0).astype(np.float32)

    if not np.isfinite(qhat).all():
        raise ValueError("crossfit_qhat produced non-finite predictions.")
    return qhat


# ============================================================================
# OPE point estimators
# ============================================================================
def compute_ope_components(
    data: BanditData,
    qhat: np.ndarray,
    switch_tau: float = cfg.SWITCH_TAU,
    weight_clip: Optional[float] = None,
) -> OPEComponents:
    n = data.n
    row = np.arange(n)
    q_obs = qhat[row, data.action]
    qpi0 = np.sum(data.pi0 * qhat, axis=1)
    qpi1 = np.sum(data.pi1 * qhat, axis=1)

    p0_obs = data.pi0[row, data.action]
    p1_obs = data.pi1[row, data.action]
    w0 = p0_obs / np.clip(data.pscore, EPS, None)
    w1 = p1_obs / np.clip(data.pscore, EPS, None)
    if weight_clip is not None:
        w0_used = np.minimum(w0, float(weight_clip))
        w1_used = np.minimum(w1, float(weight_clip))
    else:
        w0_used, w1_used = w0, w1

    resid = data.reward - q_obs
    ips0 = w0_used * data.reward
    ips1 = w1_used * data.reward
    dr0 = qpi0 + w0_used * resid
    dr1 = qpi1 + w1_used * resid
    switch0 = qpi0 + (w0_used <= switch_tau) * w0_used * resid
    switch1 = qpi1 + (w1_used <= switch_tau) * w1_used * resid

    return OPEComponents(
        qhat=qhat,
        q_obs=q_obs,
        dm0=qpi0,
        dm1=qpi1,
        ips0=ips0,
        ips1=ips1,
        dr0=dr0,
        dr1=dr1,
        switch0=switch0,
        switch1=switch1,
        w0=w0,
        w1=w1,
    )


def weighted_mean(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= EPS:
        return float("nan")
    return float(np.sum(w * x) / s)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    den = float(np.sum(w * w))
    if den <= EPS:
        return 0.0
    return float(np.sum(w) ** 2 / den)


def snips_delta(
    data: BanditData,
    mask: np.ndarray,
    base_weight: Optional[np.ndarray] = None,
    weight_clip: Optional[float] = None,
) -> float:
    idx = np.asarray(mask, dtype=bool)
    row = np.arange(data.n)
    w0 = data.pi0[row, data.action] / np.clip(data.pscore, EPS, None)
    w1 = data.pi1[row, data.action] / np.clip(data.pscore, EPS, None)
    if weight_clip is not None:
        w0 = np.minimum(w0, float(weight_clip))
        w1 = np.minimum(w1, float(weight_clip))
    r = np.ones(data.n, dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    a0 = r[idx] * w0[idx]
    a1 = r[idx] * w1[idx]
    v0 = np.sum(a0 * data.reward[idx]) / max(np.sum(a0), EPS)
    v1 = np.sum(a1 * data.reward[idx]) / max(np.sum(a1), EPS)
    return float(v1 - v0)


def point_estimates_by_group(
    data: BanditData,
    comp: OPEComponents,
    base_weight: Optional[np.ndarray] = None,
    weight_clip: Optional[float] = None,
) -> pd.DataFrame:
    base_weight = np.ones(data.n, dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    rows = []
    groups = sorted(np.unique(data.group).tolist())
    for g in groups + [-1]:
        mask = np.ones(data.n, dtype=bool) if g == -1 else (data.group == g)
        wg = base_weight[mask]
        rows.extend(
            [
                {"group": g, "estimator": "dm", "estimate": weighted_mean(comp.dm_delta[mask], wg)},
                {"group": g, "estimator": "ips", "estimate": weighted_mean(comp.ips_delta[mask], wg)},
                {"group": g, "estimator": "snips", "estimate": snips_delta(data, mask, base_weight, weight_clip)},
                {"group": g, "estimator": "dr", "estimate": weighted_mean(comp.dr_delta[mask], wg)},
                {"group": g, "estimator": "switch_dr", "estimate": weighted_mean(comp.switch_delta[mask], wg)},
            ]
        )
    return pd.DataFrame(rows)


# ============================================================================
# Uncertainty engines: ordinary bootstrap and Bayesian bootstrap
# ============================================================================
def _weighted_bootstrap_means(
    values: np.ndarray,
    base_weight: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    bayesian: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    keep = np.isfinite(values) & np.isfinite(base_weight) & (base_weight > 0)
    values = values[keep]
    base_weight = base_weight[keep]
    n = len(values)
    if n == 0:
        return np.full(n_draws, np.nan)
    if n == 1:
        return np.full(n_draws, values[0])

    out = np.empty(n_draws, dtype=float)
    # Chunking avoids allocating n_draws x n for large groups.
    chunk = max(1, min(100, n_draws))
    for start in range(0, n_draws, chunk):
        m = min(chunk, n_draws - start)
        if bayesian:
            # iid Exp(1) normalized weights are equivalent to Dirichlet(1,...,1).
            e = rng.exponential(1.0, size=(m, n))
            ww = e * base_weight[None, :]
            out[start:start + m] = (ww @ values) / np.clip(ww.sum(axis=1), EPS, None)
        else:
            # Standard nonparametric bootstrap, retaining any pre-specified base weights.
            for j in range(m):
                ii = rng.integers(0, n, size=n)
                bw = base_weight[ii]
                out[start + j] = np.sum(bw * values[ii]) / max(np.sum(bw), EPS)
    return out


def dr_uncertainty_by_group(
    z: np.ndarray,
    group: np.ndarray,
    n_draws: int,
    seed: int,
    base_weight: Optional[np.ndarray] = None,
    bayesian: bool = True,
) -> dict[int, np.ndarray]:
    z = np.asarray(z, dtype=float)
    group = np.asarray(group, dtype=int)
    base_weight = np.ones(len(z), dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for g in sorted(np.unique(group)):
        m = group == g
        out[int(g)] = _weighted_bootstrap_means(z[m], base_weight[m], n_draws, rng, bayesian)
    out[-1] = _weighted_bootstrap_means(z, base_weight, n_draws, rng, bayesian)
    return out


def snips_bootstrap_draws_by_group(
    data: BanditData,
    n_draws: int,
    seed: int,
    base_weight: Optional[np.ndarray] = None,
    weight_clip: Optional[float] = None,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_weight = np.ones(data.n, dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    row = np.arange(data.n)
    w0 = data.pi0[row, data.action] / np.clip(data.pscore, EPS, None)
    w1 = data.pi1[row, data.action] / np.clip(data.pscore, EPS, None)
    if weight_clip is not None:
        w0 = np.minimum(w0, float(weight_clip))
        w1 = np.minimum(w1, float(weight_clip))

    out = {}
    for g in sorted(np.unique(data.group).tolist()) + [-1]:
        idx = np.arange(data.n) if g == -1 else np.where(data.group == g)[0]
        n = len(idx)
        draws = np.empty(n_draws, dtype=float)
        if n == 0:
            draws[:] = np.nan
        else:
            for b in range(n_draws):
                jj = idx[rng.integers(0, n, size=n)]
                bw = base_weight[jj]
                a0 = bw * w0[jj]
                a1 = bw * w1[jj]
                v0 = np.sum(a0 * data.reward[jj]) / max(np.sum(a0), EPS)
                v1 = np.sum(a1 * data.reward[jj]) / max(np.sum(a1), EPS)
                draws[b] = v1 - v0
        out[int(g)] = draws
    return out


def summarize_draws(draws: np.ndarray, alpha: float = cfg.ALPHA, delta: float = cfg.DELTA) -> dict:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {
            "estimate_draw_mean": np.nan,
            "sd": np.nan,
            "lower_one_sided": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "interval_width": np.nan,
            "prob_gt_delta": np.nan,
            "gate": 0,
        }
    lo1 = float(np.quantile(d, alpha))
    lo2 = float(np.quantile(d, alpha / 2.0))
    hi2 = float(np.quantile(d, 1.0 - alpha / 2.0))
    return {
        "estimate_draw_mean": float(np.mean(d)),
        "sd": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        "lower_one_sided": lo1,
        "ci_lower": lo2,
        "ci_upper": hi2,
        "interval_width": hi2 - lo2,
        "prob_gt_delta": float(np.mean(d > delta)),
        "gate": int(lo1 > delta),
    }


def summaries_from_draw_dict(draw_dict: dict[int, np.ndarray], alpha: float = cfg.ALPHA, delta: float = cfg.DELTA) -> pd.DataFrame:
    rows = []
    for g, d in draw_dict.items():
        rows.append({"group": int(g), **summarize_draws(d, alpha, delta)})
    return pd.DataFrame(rows)


# ============================================================================
# Evidence diagnostics and Gate metrics
# ============================================================================
def evidence_diagnostics(
    data: BanditData,
    comp: OPEComponents,
    base_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    base_weight = np.ones(data.n, dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    rows = []
    for g in sorted(np.unique(data.group).tolist()) + [-1]:
        mask = np.ones(data.n, dtype=bool) if g == -1 else data.group == g
        w = comp.w1[mask]
        rw = base_weight[mask]
        joint = rw * w
        rows.append(
            {
                "group": int(g),
                "n_group": int(mask.sum()),
                "group_fraction": float(mask.mean()),
                "ess_pi1": effective_sample_size(w),
                "ess_fraction": effective_sample_size(w) / max(int(mask.sum()), 1),
                "weighted_ess": effective_sample_size(joint),
                "weighted_ess_fraction": effective_sample_size(joint) / max(int(mask.sum()), 1),
                "w_q95": float(np.quantile(w, 0.95)) if len(w) else np.nan,
                "w_q99": float(np.quantile(w, 0.99)) if len(w) else np.nan,
                "w_max": float(np.max(w)) if len(w) else np.nan,
                "pscore_min": float(np.min(data.pscore[mask])) if mask.any() else np.nan,
                "relevance_q05": float(np.quantile(rw, 0.05)) if len(rw) else np.nan,
                "relevance_q50": float(np.quantile(rw, 0.50)) if len(rw) else np.nan,
                "relevance_q95": float(np.quantile(rw, 0.95)) if len(rw) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_truth_metrics(records: pd.DataFrame, truth: pd.DataFrame, delta_col: str = "delta_true") -> pd.DataFrame:
    truth_small = truth[["group", delta_col]].copy()
    out = records.merge(truth_small, on="group", how="left")
    out["error"] = out["estimate"] - out[delta_col]
    out["abs_error"] = np.abs(out["error"])
    out["sq_error"] = out["error"] ** 2
    out["sign_correct"] = ((out["estimate"] > 0) == (out[delta_col] > 0)).astype(float)
    return out


def decision_metrics_from_truth(
    gate_df: pd.DataFrame,
    truth: pd.DataFrame,
    delta_threshold: float = cfg.DELTA,
    population_weight_col: str = "population_weight",
) -> dict:
    t = truth[truth["group"] >= 0].copy().set_index("group")
    g = gate_df[gate_df["group"] >= 0].copy().set_index("group")
    common = sorted(set(t.index) & set(g.index))
    if not common:
        return {}
    t = t.loc[common]
    g = g.loc[common]
    p = t[population_weight_col].to_numpy(float)
    p = p / p.sum()
    d = g["gate"].to_numpy(int)
    delta_true = t["delta_true"].to_numpy(float)
    v0 = t["v0_true"].to_numpy(float)
    v1 = t["v1_true"].to_numpy(float)
    oracle_d = (delta_true > delta_threshold).astype(int)

    coverage = float(np.sum(p * d))
    harmful = float(np.sum(p * d * (delta_true < 0)))
    false_deploy_group = float(np.mean((d == 1) & (delta_true < 0)))
    false_abstain_group = float(np.mean((d == 0) & (delta_true > delta_threshold)))
    value = float(np.sum(p * (d * v1 + (1 - d) * v0)))
    oracle_value = float(np.sum(p * (oracle_d * v1 + (1 - oracle_d) * v0)))
    baseline_value = float(np.sum(p * v0))
    candidate_value = float(np.sum(p * v1))
    return {
        "deployment_coverage": coverage,
        "harmful_exposure": harmful,
        "global_candidate_harmful_exposure": float(np.sum(p * (delta_true < 0))),
        "oracle_deployment_coverage": float(np.sum(p * oracle_d)),
        "false_deploy_group_rate": false_deploy_group,
        "false_abstain_group_rate": false_abstain_group,
        "selective_value": value,
        "oracle_selective_value": oracle_value,
        "oracle_regret": oracle_value - value,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
    }


def safety_coverage_frontier_from_draws(
    draw_dict: dict[int, np.ndarray],
    truth: pd.DataFrame,
    thresholds: Optional[list[float]] = None,
    delta_threshold: float = cfg.DELTA,
) -> pd.DataFrame:
    thresholds = thresholds or cfg.POSTERIOR_GATE_THRESHOLDS
    t = truth[truth["group"] >= 0].copy().set_index("group")
    groups = sorted(set(t.index) & {g for g in draw_dict if g >= 0})
    p = t.loc[groups, "population_weight"].to_numpy(float)
    p = p / p.sum()
    delta_true = t.loc[groups, "delta_true"].to_numpy(float)
    probs = np.array([np.mean(np.asarray(draw_dict[g]) > delta_threshold) for g in groups])
    rows = []
    for th in thresholds:
        d = (probs >= th).astype(int)
        rows.append(
            {
                "posterior_threshold": float(th),
                "deployment_coverage": float(np.sum(p * d)),
                "harmful_exposure": float(np.sum(p * d * (delta_true < 0))),
                "n_deployed_groups": int(d.sum()),
            }
        )
    return pd.DataFrame(rows)


def safe_coverage_at_eta(frontier: pd.DataFrame, eta: float) -> float:
    ok = frontier[frontier["harmful_exposure"] <= eta + 1e-15]
    return float(ok["deployment_coverage"].max()) if len(ok) else 0.0


# ============================================================================
# Normal-normal empirical-Bayes partial pooling (S2 diagnostic extension)
# ============================================================================
def empirical_bayes_partial_pooling(
    group_estimates: dict[int, float],
    group_ses: dict[int, float],
    n_draws: int,
    seed: int,
) -> dict[int, np.ndarray]:
    groups = sorted(group_estimates)
    y = np.array([group_estimates[g] for g in groups], dtype=float)
    s2 = np.array([max(group_ses[g] ** 2, 1e-10) for g in groups], dtype=float)
    # Simple empirical-Bayes moment estimate.  It is explicitly labeled EB, not exact Bayes.
    mu = float(np.average(y, weights=1.0 / s2))
    between = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    tau2 = max(between - float(np.mean(s2)), 1e-8)
    rng = np.random.default_rng(seed)
    out = {}
    for g, yg, sg2 in zip(groups, y, s2):
        post_var = 1.0 / (1.0 / sg2 + 1.0 / tau2)
        post_mean = post_var * (yg / sg2 + mu / tau2)
        out[g] = rng.normal(post_mean, math.sqrt(post_var), size=n_draws)
    return out


def complete_pooling_draws(
    z: np.ndarray,
    group: np.ndarray,
    n_draws: int,
    seed: int,
    base_weight: Optional[np.ndarray] = None,
) -> dict[int, np.ndarray]:
    global_draw = dr_uncertainty_by_group(z, np.zeros_like(group), n_draws, seed, base_weight, bayesian=True)[0]
    return {int(g): global_draw.copy() for g in np.unique(group)}


# ============================================================================
# Shift / historical relevance weighting
# ============================================================================
def estimate_density_ratio_log_to_deploy(
    X_log: np.ndarray,
    X_deploy: np.ndarray,
    seed: int,
    clip: tuple[float, float] = (0.05, 20.0),
) -> np.ndarray:
    """Estimate p_deploy(x)/p_log(x) by a domain classifier with prior correction."""
    X = np.vstack([X_log, X_deploy])
    y = np.concatenate([np.zeros(len(X_log), dtype=int), np.ones(len(X_deploy), dtype=int)])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
    )
    model.fit(X, y)
    pD = np.clip(model.predict_proba(X_log)[:, 1], 1e-6, 1 - 1e-6)
    # odds = p(D=1|x)/p(D=0|x) = p_D(x)P(D=1) / p_L(x)P(D=0)
    prior_correction = len(X_deploy) / max(len(X_log), 1)
    ratio = (pD / (1.0 - pD)) / max(prior_correction, EPS)
    return np.clip(ratio, clip[0], clip[1])


def recency_weights(
    time: np.ndarray,
    scheme: dict,
    X_log: Optional[np.ndarray] = None,
    X_deploy: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    tmax = float(np.max(time))
    tmin = float(np.min(time))
    denom = max(tmax - tmin, EPS)
    age = (tmax - time) / denom  # 0=current, 1=oldest
    kind = scheme["kind"]
    param = scheme.get("param")

    if kind == "full":
        r = np.ones_like(age)
    elif kind == "rolling":
        frac = float(param)
        if not 0 < frac <= 1:
            raise ValueError("rolling param must be in (0,1].")
        r = (age <= frac).astype(float)
    elif kind == "exp":
        half_life = float(param)
        r = np.exp(-math.log(2.0) * age / max(half_life, EPS))
    elif kind == "relevance":
        half_life = float(param)
        r_time = np.exp(-math.log(2.0) * age / max(half_life, EPS))
        if X_log is None or X_deploy is None:
            raise ValueError("relevance scheme requires X_log and X_deploy.")
        r_x = estimate_density_ratio_log_to_deploy(X_log, X_deploy, seed=seed)
        r = r_time * r_x
    else:
        raise ValueError(f"Unknown recency scheme kind={kind}")

    # Normalize mean weight to one; this changes scale, not relative weighting.
    if r.sum() <= EPS:
        raise ValueError(f"Recency scheme {scheme['name']} retained zero observations.")
    r = r / np.mean(r[r > 0])
    return r


# ============================================================================
# OBD action-context-aware linear reward model
# ============================================================================
def obd_feature_matrix(
    X: np.ndarray,
    z_action: np.ndarray,
    position: np.ndarray,
    n_positions: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    z = np.asarray(z_action, dtype=np.float32)
    pos = np.asarray(position, dtype=int)
    pos_oh = np.eye(n_positions, dtype=np.float32)[pos]
    # Context x item-feature interactions allow item effects to vary by user context.
    # We intentionally avoid an n x K item one-hot matrix so full OBD remains memory-safe.
    inter = (X[:, :, None] * z[:, None, :]).reshape(len(X), -1)
    return np.hstack([X, z, inter, pos_oh]).astype(np.float32, copy=False)

def _make_obd_sgd(seed: int):
    # partial_fit is used in the full-OBD path; max_iter is intentionally not
    # relied on because each call performs one incremental pass.
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-5,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=seed,
        average=True,
    )


def crossfit_obd_components(
    X: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    pi0_by_pos: np.ndarray,
    pi1_by_pos: np.ndarray,
    n_folds: int,
    seed: int,
    batch_size: int = 50_000,
    baseline_is_behavior: bool = False,
) -> OBDComponents:
    """Memory-safe cross-fitted OBD DR components.

    The full OBD can contain millions of rounds, so this routine deliberately
    avoids materializing one full design matrix.  For each cross-fitting fold it:
    (1) partial-fits a StandardScaler on training chunks,
    (2) partial-fits an SGD reward model on scaled training chunks, and
    (3) predicts the held-out fold in chunks.

    pi*_by_pos has shape (n_positions, n_actions).  Random/BTS are context-free
    production MAB policies, so E_pi[action_context] can be calculated once per
    position. Because the reward features are linear in action_context, prediction
    at that expected feature equals the policy-weighted model prediction.
    """
    X = np.asarray(X, dtype=np.float32)
    action = np.asarray(action, dtype=int)
    reward = np.asarray(reward, dtype=float)
    pscore = np.asarray(pscore, dtype=float)
    position = np.asarray(position, dtype=int)
    action_context = np.asarray(action_context, dtype=np.float32)
    n = len(action)
    n_actions = action_context.shape[0]
    n_positions = pi0_by_pos.shape[0]
    n_folds = max(2, min(int(n_folds), n))
    if pi0_by_pos.shape != pi1_by_pos.shape or pi0_by_pos.shape[1] != n_actions:
        raise ValueError("OBD policy-by-position matrix shape mismatch.")
    if np.max(action) >= n_actions:
        raise ValueError("OBD action index exceeds action_context rows.")
    if np.any(pscore <= 0):
        raise ValueError("OBD pscore must be positive.")

    q_obs = np.zeros(n, dtype=np.float32)
    q0 = np.zeros(n, dtype=np.float32)
    q1 = np.zeros(n, dtype=np.float32)
    ez0 = pi0_by_pos @ action_context
    ez1 = pi1_by_pos @ action_context

    # Deterministic pseudo-random fold assignment without a full permutation.
    idx64 = np.arange(n, dtype=np.int64)
    fold_id = ((idx64 * np.int64(1103515245) + np.int64(seed) * 12345) % n_folds).astype(np.int8)
    train_epochs = 2 if cfg.RUN_MODE == "paper" else 1

    for fold in range(n_folds):
        scaler = StandardScaler()
        # Pass 1: scaler on the training fold.
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            local = np.arange(start, stop)
            local = local[fold_id[start:stop] != fold]
            if len(local) == 0:
                continue
            z = action_context[action[local]]
            F = obd_feature_matrix(X[local], z, position[local], n_positions)
            scaler.partial_fit(F)

        model = _make_obd_sgd(seed + fold + 1)
        # Pass 2(+): incremental regression. Shuffle only within each chunk to
        # avoid feeding timestamp-sorted rows to SGD in a fixed local order.
        rng = np.random.default_rng(seed + 1000 + fold)
        fitted = False
        for epoch in range(train_epochs):
            for start in range(0, n, batch_size):
                stop = min(n, start + batch_size)
                local = np.arange(start, stop)
                local = local[fold_id[start:stop] != fold]
                if len(local) == 0:
                    continue
                local = local[rng.permutation(len(local))]
                z = action_context[action[local]]
                F = obd_feature_matrix(X[local], z, position[local], n_positions)
                Fs = scaler.transform(F)
                model.partial_fit(Fs, reward[local])
                fitted = True
        if not fitted:
            raise RuntimeError("OBD cross-fitting fold has no training rows.")

        # Held-out prediction pass.
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            te = np.arange(start, stop)
            te = te[fold_id[start:stop] == fold]
            if len(te) == 0:
                continue
            zobs = action_context[action[te]]
            Fobs = scaler.transform(obd_feature_matrix(X[te], zobs, position[te], n_positions))
            q_obs[te] = np.clip(model.predict(Fobs), 0.0, 1.0).astype(np.float32)

            z0 = ez0[position[te]]
            z1 = ez1[position[te]]
            F0 = scaler.transform(obd_feature_matrix(X[te], z0, position[te], n_positions))
            F1 = scaler.transform(obd_feature_matrix(X[te], z1, position[te], n_positions))
            q0[te] = np.clip(model.predict(F0), 0.0, 1.0).astype(np.float32)
            q1[te] = np.clip(model.predict(F1), 0.0, 1.0).astype(np.float32)

    p0_obs = pi0_by_pos[position, action]
    p1_obs = pi1_by_pos[position, action]
    w0 = p0_obs / np.clip(pscore, EPS, None)
    w1 = p1_obs / np.clip(pscore, EPS, None)
    resid = reward - q_obs
    dr0 = q0 + w0 * resid
    dr1 = q1 + w1 * resid
    if baseline_is_behavior:
        # When pi0 is exactly the behavior policy that generated this log,
        # its observed-action propensity ratio is identically one and
        # its DR pseudo-outcome reduces algebraically to observed reward.
        w0 = np.ones_like(w0)
        dr0 = reward.astype(float).copy()
    if not (np.isfinite(dr0).all() and np.isfinite(dr1).all()):
        raise ValueError("OBD DR components contain non-finite values.")
    return OBDComponents(q_obs=q_obs, q_pi0=q0, q_pi1=q1, dr0=dr0, dr1=dr1, w0=w0, w1=w1)


# ============================================================================
# I/O and plotting
# ============================================================================
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {path}")


def save_show(fig: plt.Figure, path: Path, show: Optional[bool] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"[figure] {path}")
    if cfg.SHOW_FIGURES if show is None else show:
        plt.show()
    else:
        plt.close(fig)


def print_frame(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total; full table saved to disk)")


def plot_safety_coverage(frontiers: pd.DataFrame, title: str, path: Path, label_col: str = "method") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, d in frontiers.groupby(label_col, dropna=False):
        d = d.sort_values("deployment_coverage")
        ax.plot(d["deployment_coverage"], d["harmful_exposure"], marker="o", label=str(label))
    ax.set_xlabel("Deployment coverage")
    ax.set_ylabel("Harmful exposure")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_show(fig, path)


def plot_calibration(calib: pd.DataFrame, title: str, path: Path, hue: str = "method") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, d in calib.groupby(hue, dropna=False):
        ax.plot(d["nominal"], d["empirical"], marker="o", label=str(label))
    x = np.linspace(0.5, 1.0, 100)
    ax.plot(x, x, linestyle="--", linewidth=1.0, label="ideal")
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_show(fig, path)


def monte_carlo_summary(records: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, d in records.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            x = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(x)) if len(x) else np.nan
            row[f"{col}_mcse"] = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan
        row["n_rep"] = len(d)
        rows.append(row)
    return pd.DataFrame(rows)
