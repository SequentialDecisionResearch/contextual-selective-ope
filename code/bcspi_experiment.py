"""Reusable experiment evaluator for BC-SPI Synthetic and Semi-Synthetic runs.

The dataset-specific runners create BanditData + evaluator-only truth.  This file
then performs the SAME cross-fitted nuisance fitting, OPE, uncertainty, Gate,
calibration, evidence diagnostics and decision metrics across those two layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

import bcspi_config as cfg
from bcspi_core import (
    BanditData,
    OPEComponents,
    crossfit_qhat,
    compute_ope_components,
    point_estimates_by_group,
    dr_uncertainty_by_group,
    snips_bootstrap_draws_by_group,
    summaries_from_draw_dict,
    evidence_diagnostics,
    decision_metrics_from_truth,
    safety_coverage_frontier_from_draws,
    safe_coverage_at_eta,
)


@dataclass
class GenericEvaluation:
    point: pd.DataFrame
    gates: pd.DataFrame
    decisions: pd.DataFrame
    calibration: pd.DataFrame
    frontiers: pd.DataFrame
    diagnostics: pd.DataFrame
    qhat: np.ndarray
    components: OPEComponents
    draws: dict[str, dict[int, np.ndarray]]


def _tag(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    for k, v in reversed(list(meta.items())):
        out.insert(0, k, v)
    return out


def _interval_from_draws(draws: np.ndarray, nominal: float) -> tuple[float, float]:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    tail = (1.0 - nominal) / 2.0
    return float(np.quantile(d, tail)), float(np.quantile(d, 1.0 - tail))


def _snips_asymptotic_gate(data: BanditData, alpha: float, delta: float, base_weight: np.ndarray) -> pd.DataFrame:
    """Asymptotic self-normalized (SNIPS) LCB baseline.

    This is intentionally labelled asymptotic, not a finite-sample guarantee.
    The influence-function approximation accounts for the ratio normalization.
    """
    row = np.arange(data.n)
    w0 = data.pi0[row, data.action] / np.clip(data.pscore, 1e-12, None)
    w1 = data.pi1[row, data.action] / np.clip(data.pscore, 1e-12, None)
    zcrit = float(norm.ppf(1.0 - alpha))
    rows = []
    for g in sorted(np.unique(data.group).tolist()) + [-1]:
        mask = np.ones(data.n, dtype=bool) if g == -1 else data.group == g
        r = base_weight[mask]
        y = data.reward[mask]
        a0 = r * w0[mask]
        a1 = r * w1[mask]
        s0 = max(float(np.sum(a0)), 1e-12)
        s1 = max(float(np.sum(a1)), 1e-12)
        v0 = float(np.sum(a0 * y) / s0)
        v1 = float(np.sum(a1 * y) / s1)
        est = v1 - v0
        # Ratio-estimator linearization.  Scaling by mean denominator makes the
        # sample mean of IF approximate the estimator perturbation.
        n = int(mask.sum())
        if n <= 1:
            se = np.nan
        else:
            mean_a0 = max(float(np.mean(a0)), 1e-12)
            mean_a1 = max(float(np.mean(a1)), 1e-12)
            infl = a1 * (y - v1) / mean_a1 - a0 * (y - v0) / mean_a0
            se = float(np.std(infl, ddof=1) / np.sqrt(n))
        lower = est - zcrit * se if np.isfinite(se) else np.nan
        rows.append({
            "group": int(g), "estimate_draw_mean": est, "sd": se,
            "lower_one_sided": lower, "ci_lower": est - 1.96 * se if np.isfinite(se) else np.nan,
            "ci_upper": est + 1.96 * se if np.isfinite(se) else np.nan,
            "interval_width": 3.92 * se if np.isfinite(se) else np.nan,
            "prob_gt_delta": float(norm.cdf((est - delta) / se)) if np.isfinite(se) and se > 0 else float(est > delta),
            "gate": int(np.isfinite(lower) and lower > delta),
        })
    return pd.DataFrame(rows)


def evaluate_generic_bandit(
    data: BanditData,
    truth: pd.DataFrame,
    meta: dict,
    model_kind: str,
    seed: int,
    base_weight: Optional[np.ndarray] = None,
    bootstrap_draws: Optional[int] = None,
    bayes_draws: Optional[int] = None,
    n_folds: Optional[int] = None,
    include_snips: bool = True,
) -> GenericEvaluation:
    """Run the core BC-SPI evaluation on one realized logged dataset."""
    base_weight = np.ones(data.n, dtype=float) if base_weight is None else np.asarray(base_weight, dtype=float)
    bootstrap_draws = int(bootstrap_draws or cfg.BOOTSTRAP_DRAWS)
    bayes_draws = int(bayes_draws or cfg.BAYES_BOOTSTRAP_DRAWS)
    n_folds = int(n_folds or cfg.N_FOLDS)

    qhat = crossfit_qhat(data, model_kind=model_kind, n_folds=n_folds, seed=seed, fit_weight=base_weight)
    comp = compute_ope_components(data, qhat)

    point = point_estimates_by_group(data, comp, base_weight=base_weight)
    point = point.merge(truth[["group", "delta_true"]], on="group", how="left")
    point["error"] = point["estimate"] - point["delta_true"]
    point["abs_error"] = point["error"].abs()
    point["sq_error"] = point["error"] ** 2
    point = _tag(point, {**meta, "nuisance_model": model_kind})

    draws: dict[str, dict[int, np.ndarray]] = {}
    draws["bootstrap_dr"] = dr_uncertainty_by_group(
        comp.dr_delta, data.group, bootstrap_draws, seed + 11, base_weight, bayesian=False
    )
    draws["bb_dr"] = dr_uncertainty_by_group(
        comp.dr_delta, data.group, bayes_draws, seed + 17, base_weight, bayesian=True
    )
    draws["bb_dm"] = dr_uncertainty_by_group(
        comp.dm_delta, data.group, bayes_draws, seed + 23, base_weight, bayesian=True
    )
    if include_snips:
        draws["bootstrap_snips"] = snips_bootstrap_draws_by_group(
            data, bootstrap_draws, seed + 29, base_weight=base_weight
        )

    diag = evidence_diagnostics(data, comp, base_weight=base_weight)
    diag = _tag(diag, {**meta, "nuisance_model": model_kind})
    diag_lookup = diag.set_index("group")

    gate_frames = []
    decision_rows = []
    frontier_frames = []
    calibration_rows = []

    # Draw-based methods.
    for method, dd in draws.items():
        s = summaries_from_draw_dict(dd, alpha=cfg.ALPHA, delta=cfg.DELTA)
        s.insert(0, "method", method)
        s = s.merge(truth[["group", "delta_true"]], on="group", how="left")
        # Attach diagnostics used to interpret the Gate.
        s = s.merge(
            diag[["group", "n_group", "group_fraction", "ess_pi1", "ess_fraction", "weighted_ess", "weighted_ess_fraction", "w_q95", "w_q99", "w_max", "pscore_min"]],
            on="group", how="left"
        )
        s = _tag(s, {**meta, "nuisance_model": model_kind})
        gate_frames.append(s)

        dm = decision_metrics_from_truth(s[["group", "gate"]], truth)
        decision_rows.append({**meta, "nuisance_model": model_kind, "method": method, **dm})

        fr = safety_coverage_frontier_from_draws(dd, truth)
        fr.insert(0, "method", method)
        for eta in cfg.SAFE_ETA_GRID:
            fr[f"safe_coverage_eta_{eta:g}"] = safe_coverage_at_eta(fr, eta)
        frontier_frames.append(_tag(fr, {**meta, "nuisance_model": model_kind}))

        for nominal in cfg.COVERAGE_LEVELS:
            for g, gd in dd.items():
                tr = truth.loc[truth["group"] == g, "delta_true"]
                if tr.empty:
                    continue
                lo, hi = _interval_from_draws(gd, nominal)
                tv = float(tr.iloc[0])
                calibration_rows.append({
                    **meta, "nuisance_model": model_kind, "method": method,
                    "group": int(g), "nominal": float(nominal),
                    "covered": int(np.isfinite(lo) and lo <= tv <= hi),
                    "interval_lower": lo, "interval_upper": hi,
                    "interval_width": hi - lo if np.isfinite(lo + hi) else np.nan,
                })

    # ESS-safeguarded BB-DR ablation.
    bb_s = summaries_from_draw_dict(draws["bb_dr"], alpha=cfg.ALPHA, delta=cfg.DELTA)
    bb_s["gate"] = [
        int(row.gate == 1 and row.group in diag_lookup.index and float(diag_lookup.loc[row.group, "ess_fraction"]) >= cfg.ESS_SAFEGUARD_FRAC)
        if int(row.group) >= 0 else int(row.gate)
        for row in bb_s.itertuples(index=False)
    ]
    bb_s.insert(0, "method", "bb_dr_ess_safeguard")
    bb_s = bb_s.merge(truth[["group", "delta_true"]], on="group", how="left")
    bb_s = bb_s.merge(
        diag[["group", "n_group", "group_fraction", "ess_pi1", "ess_fraction", "weighted_ess", "weighted_ess_fraction", "w_q95", "w_q99", "w_max", "pscore_min"]],
        on="group", how="left"
    )
    bb_s = _tag(bb_s, {**meta, "nuisance_model": model_kind})
    gate_frames.append(bb_s)
    decision_rows.append({
        **meta, "nuisance_model": model_kind, "method": "bb_dr_ess_safeguard",
        **decision_metrics_from_truth(bb_s[["group", "gate"]], truth)
    })

    # Asymptotic self-normalized policy-selection baseline.
    sn = _snips_asymptotic_gate(data, cfg.ALPHA, cfg.DELTA, base_weight)
    sn.insert(0, "method", "snips_asym_lcb")
    sn = sn.merge(truth[["group", "delta_true"]], on="group", how="left")
    sn = sn.merge(
        diag[["group", "n_group", "group_fraction", "ess_pi1", "ess_fraction", "weighted_ess", "weighted_ess_fraction", "w_q95", "w_q99", "w_max", "pscore_min"]],
        on="group", how="left"
    )
    sn = _tag(sn, {**meta, "nuisance_model": model_kind})
    gate_frames.append(sn)
    decision_rows.append({
        **meta, "nuisance_model": model_kind, "method": "snips_asym_lcb",
        **decision_metrics_from_truth(sn[["group", "gate"]], truth)
    })

    gates = pd.concat(gate_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    calibration = pd.DataFrame(calibration_rows)
    frontiers = pd.concat(frontier_frames, ignore_index=True) if frontier_frames else pd.DataFrame()

    # Explicit policy reference points make the final Safety-Coverage plot interpretable:
    # never deploy candidate, deploy candidate everywhere, and oracle selective deployment.
    tt = truth[truth["group"] >= 0].copy()
    if len(tt):
        pp = tt["population_weight"].to_numpy(float); pp = pp / pp.sum()
        dt = tt["delta_true"].to_numpy(float)
        refs = pd.DataFrame([
            {"method": "always_baseline", "posterior_threshold": np.nan, "deployment_coverage": 0.0, "harmful_exposure": 0.0, "n_deployed_groups": 0},
            {"method": "global_candidate", "posterior_threshold": np.nan, "deployment_coverage": 1.0, "harmful_exposure": float(np.sum(pp * (dt < 0))), "n_deployed_groups": int(len(tt))},
            {"method": "oracle_selective", "posterior_threshold": np.nan, "deployment_coverage": float(np.sum(pp * (dt > cfg.DELTA))), "harmful_exposure": 0.0, "n_deployed_groups": int(np.sum(dt > cfg.DELTA))},
        ])
        frontiers = pd.concat([frontiers, _tag(refs, {**meta, "nuisance_model": model_kind})], ignore_index=True)

    return GenericEvaluation(
        point=point,
        gates=gates,
        decisions=decisions,
        calibration=calibration,
        frontiers=frontiers,
        diagnostics=diag,
        qhat=qhat,
        components=comp,
        draws=draws,
    )
