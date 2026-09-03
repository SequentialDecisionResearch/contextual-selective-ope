"""
BC-SPI targeted OBD DR audit
============================

Purpose
-------
Diagnose the ONE remaining real-log issue identified by final_article_analysis.py:
why OBD DR / BB-DR can disagree strongly with the independent on-policy reference,
especially for the BTS -> Random negative control.

This is intentionally a SMALL, targeted diagnostic. It does NOT rerun Synthetic or
Semi-Synthetic, and it does NOT rerun the full OBD paper grid.

Default audit scope
-------------------
- campaign: all
- one deterministic 50,000-row behavior sample per direction
- exactly the same sample seed as OBD paper run, rep=0
- current squared-error SGD nuisance (same structure as bcspi_core)
- explicit per-action q_pi audit (tests the expected-action-context shortcut)
- pre-specified weight clipping at SWITCH_TAU (default 10)
- one alternative logistic-loss SGD nuisance
- behavior-policy identity check pi0(A|position) / logged pscore
- comparison with existing independent on-policy reference and saved OBD 50k result

Expected project layout
-----------------------
Place this file in:
    C:\\study_notes\\traval_rec\\Contextual_bandit_bayesian_ope_safe_imp\\BCSPI_python_code

The project root must contain:
    bcspi_config.py
    bcspi_core.py
    run_obdnew.py

Existing paper results must contain:
    bcspi_results\\tables\\obd_onpolicy_reference.csv
    bcspi_results\\tables\\obd_point_estimates_long.csv

Full OBD data path is read from bcspi_config.OBD_DATA_PATH.

Spyder
------
Open this file in Spyder and press F5.

Outputs
-------
Written ONLY under:
    bcspi_results\\obd_dr_audit

Files:
    audit_behavior_identity.csv
    audit_qpi_shortcut.csv
    audit_weight_diagnostics.csv
    audit_estimator_reference_comparison.csv
    audit_saved_result_reproduction.csv
    audit_summary.csv
    audit_conclusion.txt
    audit_manifest.json
    fig_audit_estimator_mae.png
    fig_audit_behavior_identity.png

Important interpretation
------------------------
The audit does not treat the OBD independent on-policy log as oracle truth. Clear
reference groups are used only where the existing Newcombe/Wilson interval excludes
zero, exactly as in the paper pipeline.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

import bcspi_config as cfg
from bcspi_core import obd_feature_matrix


# =============================================================================
# Small, frozen audit settings
# =============================================================================

AUDIT_CAMPAIGNS = ["all"]
AUDIT_N = 50_000
AUDIT_REP = 0
AUDIT_BATCH_SIZE = 50_000
AUDIT_PREDICT_CHUNK = 5_000
AUDIT_N_FOLDS = int(getattr(cfg, "OBD_N_FOLDS", 2))
AUDIT_TRAIN_EPOCHS = 2  # paper OBD core uses 2 epochs
WEIGHT_CLIP = float(getattr(cfg, "SWITCH_TAU", 10.0))
RUN_LOGISTIC_NUISANCE = True

EPS = 1e-12
PROJECT_DIR = Path(cfg.PROJECT_DIR)
RESULTS_DIR = Path(cfg.RESULTS_DIR)
TABLES_DIR = Path(cfg.TABLES_DIR)
OUT_DIR = RESULTS_DIR / "obd_dr_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Safe loading of run_obdnew.py helpers
# =============================================================================

def load_obd_runner_helpers():
    """Load helper functions from run_obdnew.py without a normal import.

    This helper loads run_obdnew.py under a non-__main__ module name so its
    helper functions can be audited without executing the full OBD run.

    This function:
      1) reads run_obdnew.py as text;
      2) removes the future-import line in memory for dynamic compilation;
      3) compiles the remaining source under a non-__main__ module name;
      4) returns the helper namespace.

    It DOES NOT modify run_obdnew.py on disk and DOES NOT execute run_obd().
    """
    import types

    runner_path = PROJECT_DIR / "run_obdnew.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"run_obdnew.py not found: {runner_path}")

    source = runner_path.read_text(encoding="utf-8-sig")
    cleaned = "\n".join(
        line for line in source.splitlines()
        if line.strip() != "from __future__ import annotations"
    ) + "\n"

    module = types.ModuleType("_bcspi_obd_runner_for_audit")
    module.__file__ = str(runner_path)
    module.__name__ = "_bcspi_obd_runner_for_audit"

    try:
        exec(compile(cleaned, str(runner_path), "exec"), module.__dict__)
    except Exception as e:
        raise RuntimeError(
            "Could not safely load helper definitions from run_obdnew.py. "
            f"Original error: {e!r}"
        ) from e

    required = ["_load_pair"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "run_obdnew.py loaded but required helper(s) are missing: "
            + ", ".join(missing)
        )
    return module


# =============================================================================
# I/O helpers
# =============================================================================

def save_csv(df: pd.DataFrame, name: str) -> Path:
    p = OUT_DIR / name
    df.to_csv(p, index=False)
    print(f"[saved] {p}")
    return p


def save_json(obj: dict, name: str) -> Path:
    p = OUT_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {p}")
    return p


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if len(w) == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return (s1 * s1) / max(s2, EPS)


def qstats(x: np.ndarray, prefix: str = "") -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            f"{prefix}mean": np.nan,
            f"{prefix}sd": np.nan,
            f"{prefix}q01": np.nan,
            f"{prefix}q05": np.nan,
            f"{prefix}q50": np.nan,
            f"{prefix}q95": np.nan,
            f"{prefix}q99": np.nan,
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
        }
    return {
        f"{prefix}mean": float(np.mean(x)),
        f"{prefix}sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}q01": float(np.quantile(x, 0.01)),
        f"{prefix}q05": float(np.quantile(x, 0.05)),
        f"{prefix}q50": float(np.quantile(x, 0.50)),
        f"{prefix}q95": float(np.quantile(x, 0.95)),
        f"{prefix}q99": float(np.quantile(x, 0.99)),
        f"{prefix}min": float(np.min(x)),
        f"{prefix}max": float(np.max(x)),
    }


# =============================================================================
# Exact grouping model used by run_obdnew.py, but predict only audit rows
# =============================================================================

def fit_group_model_exact(
    X_behavior: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, MiniBatchKMeans]:
    """Reproduce the frozen reward-blind cluster fit from run_obdnew.py.

    We fit on the same deterministic design sample, but unlike _fit_groups we do
    not predict cluster labels for millions of rows that are not used in this audit.
    """
    X_behavior = np.asarray(X_behavior, dtype=np.float32)
    if X_behavior.ndim != 2:
        raise ValueError("OBD grouping requires a 2D context matrix.")
    if len(X_behavior) < int(cfg.OBD_CLUSTER_K):
        raise ValueError("Not enough rows for OBD clustering.")

    n_design = min(len(X_behavior), int(cfg.OBD_CLUSTER_DESIGN_MAX_N))
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(X_behavior), size=n_design, replace=False)
        if n_design < len(X_behavior)
        else np.arange(len(X_behavior))
    )

    scaler = StandardScaler()
    Xd = scaler.fit_transform(X_behavior[idx])
    km = MiniBatchKMeans(
        n_clusters=int(cfg.OBD_CLUSTER_K),
        random_state=seed,
        batch_size=4096,
        n_init=10,
    )
    km.fit(Xd)
    return scaler, km


def predict_groups(
    X: np.ndarray,
    position: np.ndarray,
    scaler: StandardScaler,
    km: MiniBatchKMeans,
    len_list: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    position = np.asarray(position, dtype=int)
    cluster = km.predict(scaler.transform(X))
    return (cluster * int(len_list) + position).astype(int)


# =============================================================================
# Current OBD nuisance model + explicit-action audit
# =============================================================================

def make_current_regressor(seed: int) -> SGDRegressor:
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-5,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=seed,
        average=True,
    )


def make_logistic_model(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        random_state=seed,
        average=True,
    )


def _predict_prob(model, Fs: np.ndarray, model_kind: str) -> np.ndarray:
    if model_kind == "current_sgd":
        return np.clip(model.predict(Fs), 0.0, 1.0)
    if model_kind == "logistic_sgd":
        return np.clip(model.predict_proba(Fs)[:, 1], 0.0, 1.0)
    raise ValueError(f"Unknown model_kind={model_kind!r}")


def explicit_policy_prediction(
    model,
    scaler: StandardScaler,
    X: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    pi_by_pos: np.ndarray,
    model_kind: str,
    chunk_size: int = AUDIT_PREDICT_CHUNK,
) -> np.ndarray:
    """Compute sum_a pi(a|position) q_hat(x,a) AFTER per-action clipping.

    This is the direct action enumeration used to test whether the current
    expected-action-context shortcut changes q_pi because clipping is nonlinear.
    """
    X = np.asarray(X, dtype=np.float32)
    position = np.asarray(position, dtype=int)
    action_context = np.asarray(action_context, dtype=np.float32)
    pi_by_pos = np.asarray(pi_by_pos, dtype=float)

    n = len(X)
    k = action_context.shape[0]
    L = pi_by_pos.shape[0]
    if pi_by_pos.shape[1] != k:
        raise ValueError("pi/action_context dimension mismatch.")

    out = np.zeros(n, dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        Xc = X[start:stop]
        pc = position[start:stop]
        acc = np.zeros(stop - start, dtype=float)
        for a in range(k):
            za = np.broadcast_to(action_context[a], (stop - start, action_context.shape[1]))
            F = obd_feature_matrix(Xc, za, pc, L)
            Fs = scaler.transform(F)
            qa = _predict_prob(model, Fs, model_kind)
            acc += pi_by_pos[pc, a] * qa
        out[start:stop] = acc
    return out


def audit_crossfit(
    X: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    pi0_by_pos: np.ndarray,
    pi1_by_pos: np.ndarray,
    seed: int,
    model_kind: str,
) -> dict[str, np.ndarray]:
    """Cross-fit one nuisance model and return both shortcut and explicit q_pi.

    For model_kind='current_sgd', training, fold assignment, scaler, SGD settings,
    and shortcut prediction reproduce bcspi_core.crossfit_obd_components.
    """
    X = np.asarray(X, dtype=np.float32)
    action = np.asarray(action, dtype=int)
    reward = np.asarray(reward, dtype=float)
    pscore = np.asarray(pscore, dtype=float)
    position = np.asarray(position, dtype=int)
    action_context = np.asarray(action_context, dtype=np.float32)
    pi0_by_pos = np.asarray(pi0_by_pos, dtype=float)
    pi1_by_pos = np.asarray(pi1_by_pos, dtype=float)

    n = len(action)
    k = action_context.shape[0]
    L = pi0_by_pos.shape[0]
    n_folds = max(2, min(int(AUDIT_N_FOLDS), n))

    if pi0_by_pos.shape != pi1_by_pos.shape or pi0_by_pos.shape[1] != k:
        raise ValueError("Policy matrix shape mismatch.")
    if np.any(pscore <= 0):
        raise ValueError("pscore must be positive.")

    q_obs = np.zeros(n, dtype=float)
    q0_short = np.zeros(n, dtype=float)
    q1_short = np.zeros(n, dtype=float)
    q0_exp = np.zeros(n, dtype=float)
    q1_exp = np.zeros(n, dtype=float)

    ez0 = pi0_by_pos @ action_context
    ez1 = pi1_by_pos @ action_context

    idx64 = np.arange(n, dtype=np.int64)
    fold_id = (
        (idx64 * np.int64(1103515245) + np.int64(seed) * 12345) % n_folds
    ).astype(np.int8)

    for fold in range(n_folds):
        scaler = StandardScaler()

        # Same scaler pass as the paper OBD core.
        for start in range(0, n, AUDIT_BATCH_SIZE):
            stop = min(n, start + AUDIT_BATCH_SIZE)
            local = np.arange(start, stop)
            local = local[fold_id[start:stop] != fold]
            if len(local) == 0:
                continue
            z = action_context[action[local]]
            F = obd_feature_matrix(X[local], z, position[local], L)
            scaler.partial_fit(F)

        if model_kind == "current_sgd":
            model = make_current_regressor(seed + fold + 1)
        elif model_kind == "logistic_sgd":
            model = make_logistic_model(seed + fold + 1)
        else:
            raise ValueError(model_kind)

        rng = np.random.default_rng(seed + 1000 + fold)
        fitted = False
        for _epoch in range(AUDIT_TRAIN_EPOCHS):
            for start in range(0, n, AUDIT_BATCH_SIZE):
                stop = min(n, start + AUDIT_BATCH_SIZE)
                local = np.arange(start, stop)
                local = local[fold_id[start:stop] != fold]
                if len(local) == 0:
                    continue
                local = local[rng.permutation(len(local))]
                z = action_context[action[local]]
                F = obd_feature_matrix(X[local], z, position[local], L)
                Fs = scaler.transform(F)
                if model_kind == "current_sgd":
                    model.partial_fit(Fs, reward[local])
                else:
                    if not fitted:
                        model.partial_fit(Fs, reward[local].astype(int), classes=np.array([0, 1]))
                    else:
                        model.partial_fit(Fs, reward[local].astype(int))
                fitted = True
        if not fitted:
            raise RuntimeError("Audit cross-fit fold has no training rows.")

        te = np.where(fold_id == fold)[0]
        for start in range(0, len(te), AUDIT_PREDICT_CHUNK):
            ids = te[start : start + AUDIT_PREDICT_CHUNK]
            zobs = action_context[action[ids]]
            Fobs = scaler.transform(obd_feature_matrix(X[ids], zobs, position[ids], L))
            q_obs[ids] = _predict_prob(model, Fobs, model_kind)

            if model_kind == "current_sgd":
                z0 = ez0[position[ids]]
                z1 = ez1[position[ids]]
                F0 = scaler.transform(obd_feature_matrix(X[ids], z0, position[ids], L))
                F1 = scaler.transform(obd_feature_matrix(X[ids], z1, position[ids], L))
                q0_short[ids] = _predict_prob(model, F0, model_kind)
                q1_short[ids] = _predict_prob(model, F1, model_kind)

            q0_exp[ids] = explicit_policy_prediction(
                model, scaler, X[ids], position[ids], action_context,
                pi0_by_pos, model_kind, AUDIT_PREDICT_CHUNK,
            )
            q1_exp[ids] = explicit_policy_prediction(
                model, scaler, X[ids], position[ids], action_context,
                pi1_by_pos, model_kind, AUDIT_PREDICT_CHUNK,
            )

    if model_kind != "current_sgd":
        # Logistic model has no valid linear expected-feature shortcut. Make the
        # explicit prediction its only q_pi definition.
        q0_short[:] = q0_exp
        q1_short[:] = q1_exp

    p0_obs = pi0_by_pos[position, action]
    p1_obs = pi1_by_pos[position, action]
    w0 = p0_obs / np.clip(pscore, EPS, None)
    w1 = p1_obs / np.clip(pscore, EPS, None)

    return {
        "q_obs": q_obs,
        "q0_short": q0_short,
        "q1_short": q1_short,
        "q0_explicit": q0_exp,
        "q1_explicit": q1_exp,
        "w0": w0,
        "w1": w1,
    }


# =============================================================================
# Estimators and comparison with independent on-policy reference
# =============================================================================

def estimator_rows(
    comp: dict[str, np.ndarray],
    group: np.ndarray,
    reward: np.ndarray,
    model_prefix: str,
) -> pd.DataFrame:
    group = np.asarray(group, dtype=int)
    reward = np.asarray(reward, dtype=float)
    qobs = comp["q_obs"]
    w0 = comp["w0"]
    w1 = comp["w1"]
    w0c = np.minimum(w0, WEIGHT_CLIP)
    w1c = np.minimum(w1, WEIGHT_CLIP)

    q0s = comp["q0_short"]
    q1s = comp["q1_short"]
    q0e = comp["q0_explicit"]
    q1e = comp["q1_explicit"]

    rows = []
    groups = sorted(np.unique(group).tolist()) + [-1]
    for g in groups:
        m = np.ones(len(group), dtype=bool) if g == -1 else group == g
        r = reward[m]

        ips = np.mean(w1[m] * r - w0[m] * r)
        snips = (
            np.sum(w1[m] * r) / max(np.sum(w1[m]), EPS)
            - np.sum(w0[m] * r) / max(np.sum(w0[m]), EPS)
        )
        snips_clip = (
            np.sum(w1c[m] * r) / max(np.sum(w1c[m]), EPS)
            - np.sum(w0c[m] * r) / max(np.sum(w0c[m]), EPS)
        )

        dm_short = np.mean(q1s[m] - q0s[m])
        dm_exp = np.mean(q1e[m] - q0e[m])
        dr_short = np.mean((q1s[m] + w1[m] * (r - qobs[m])) - (q0s[m] + w0[m] * (r - qobs[m])))
        dr_exp = np.mean((q1e[m] + w1[m] * (r - qobs[m])) - (q0e[m] + w0[m] * (r - qobs[m])))
        dr_short_clip = np.mean((q1s[m] + w1c[m] * (r - qobs[m])) - (q0s[m] + w0c[m] * (r - qobs[m])))
        dr_exp_clip = np.mean((q1e[m] + w1c[m] * (r - qobs[m])) - (q0e[m] + w0c[m] * (r - qobs[m])))

        vals = {
            f"{model_prefix}_dm_shortcut": dm_short,
            f"{model_prefix}_dm_explicit": dm_exp,
            f"{model_prefix}_dr_shortcut_raw": dr_short,
            f"{model_prefix}_dr_explicit_raw": dr_exp,
            f"{model_prefix}_dr_shortcut_clip{WEIGHT_CLIP:g}": dr_short_clip,
            f"{model_prefix}_dr_explicit_clip{WEIGHT_CLIP:g}": dr_exp_clip,
        }
        if model_prefix == "current":
            vals.update({
                "ips_raw": ips,
                "snips_raw": snips,
                f"snips_clip{WEIGHT_CLIP:g}": snips_clip,
            })

        for estimator, value in vals.items():
            rows.append({"group": int(g), "estimator": estimator, "estimate": float(value)})

    return pd.DataFrame(rows)


def compare_to_reference(
    estimates: pd.DataFrame,
    reference: pd.DataFrame,
    campaign: str,
    direction: str,
) -> pd.DataFrame:
    ref = reference[
        reference["campaign"].eq(campaign)
        & reference["direction"].eq(direction)
    ][["group", "delta_ref", "reference_status", "population_weight"]].copy()

    z = estimates.merge(ref, on="group", how="left")
    z["error_vs_reference"] = z["estimate"] - z["delta_ref"]
    z["abs_error_vs_reference"] = z["error_vs_reference"].abs()
    z["sign_agreement"] = ((z["estimate"] > 0) == (z["delta_ref"] > 0)).astype(float)
    z["campaign"] = campaign
    z["direction"] = direction
    return z


# =============================================================================
# Mock self-test: prove the current-SGD audit reproduces bcspi_core exactly
# =============================================================================

def mock_self_test() -> None:
    print("[SELF-TEST] checking audit cross-fit against bcspi_core...")
    from bcspi_core import crossfit_obd_components

    rng = np.random.default_rng(12345)
    n = 800
    dx = 4
    k = 7
    dz = 3
    L = 3

    X = rng.normal(size=(n, dx)).astype(np.float32)
    ac = rng.normal(size=(k, dz)).astype(np.float32)
    pos = rng.integers(0, L, size=n)
    pi0 = np.full((L, k), 1.0 / k)
    raw = rng.gamma(2.0, 1.0, size=(L, k))
    pi1 = raw / raw.sum(axis=1, keepdims=True)
    action = np.array([rng.choice(k, p=pi0[p]) for p in pos], dtype=int)
    pscore = pi0[pos, action]
    logits = 0.2 * X[:, 0] + 0.15 * ac[action, 0] - 0.1 * pos
    prob = 1.0 / (1.0 + np.exp(-logits))
    reward = rng.binomial(1, prob).astype(float)
    seed = 9876

    old_n = globals()["AUDIT_N_FOLDS"]
    try:
        globals()["AUDIT_N_FOLDS"] = 2
        ours = audit_crossfit(X, action, reward, pscore, pos, ac, pi0, pi1, seed, "current_sgd")
        core = crossfit_obd_components(X, action, reward, pscore, pos, ac, pi0, pi1, 2, seed)
    finally:
        globals()["AUDIT_N_FOLDS"] = old_n

    checks = {
        "q_obs": np.max(np.abs(ours["q_obs"] - np.asarray(core.q_obs))),
        "q0_short": np.max(np.abs(ours["q0_short"] - np.asarray(core.q_pi0))),
        "q1_short": np.max(np.abs(ours["q1_short"] - np.asarray(core.q_pi1))),
        "w0": np.max(np.abs(ours["w0"] - np.asarray(core.w0))),
        "w1": np.max(np.abs(ours["w1"] - np.asarray(core.w1))),
    }
    worst = max(checks.values())
    print("  max absolute differences:", checks)
    if worst > 2e-6:
        raise AssertionError(
            f"Audit implementation does not reproduce bcspi_core closely enough; max diff={worst:.3g}"
        )
    print("[SELF-TEST PASS] current-SGD audit matches bcspi_core.")


# =============================================================================
# One direction audit
# =============================================================================

def direction_seed(campaign_index: int, direction: str, n: int, rep: int) -> int:
    return (
        int(cfg.RANDOM_SEED)
        + campaign_index * 1_000_000
        + (0 if direction == "random_to_bts" else 500_000)
        + int(n)
        + int(rep)
    )


def run_direction(
    campaign: str,
    campaign_index: int,
    direction: str,
    fr: dict,
    fb: dict,
    random_pos: np.ndarray,
    bts_pos: np.ndarray,
    group_scaler: StandardScaler,
    group_km: MiniBatchKMeans,
    len_list: int,
    reference: pd.DataFrame,
    saved_points: pd.DataFrame,
) -> dict:
    if direction == "random_to_bts":
        behavior = fr
        pi0, pi1 = random_pos, bts_pos
    elif direction == "bts_to_random":
        behavior = fb
        pi0, pi1 = bts_pos, random_pos
    else:
        raise ValueError(direction)

    n_total = int(behavior["n_rounds"])
    n = min(int(AUDIT_N), n_total)
    seed = direction_seed(campaign_index, direction, n, AUDIT_REP)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total) if n == n_total else rng.choice(n_total, size=n, replace=False)

    X = np.asarray(behavior["context"], dtype=np.float32)[idx]
    action = np.asarray(behavior["action"], dtype=int)[idx]
    reward = np.asarray(behavior["reward"], dtype=float)[idx]
    pscore = np.asarray(behavior["pscore"], dtype=float)[idx]
    position = np.asarray(behavior["position"], dtype=int)[idx]
    action_context = np.asarray(behavior["action_context"], dtype=np.float32)

    group = predict_groups(X, position, group_scaler, group_km, len_list)

    print(f"\n[AUDIT] {campaign} {direction}: n={n:,}, seed={seed}")

    # -------------------------------------------------------------------------
    # 1) Behavior-policy identity audit
    # -------------------------------------------------------------------------
    p0_reconstructed = pi0[position, action]
    rho0 = p0_reconstructed / np.clip(pscore, EPS, None)
    abs_dev = np.abs(rho0 - 1.0)
    identity = {
        "campaign": campaign,
        "direction": direction,
        "n": n,
        **qstats(rho0, "rho0_"),
        "mean_abs_deviation_from_1": float(np.mean(abs_dev)),
        "p95_abs_deviation_from_1": float(np.quantile(abs_dev, 0.95)),
        "max_abs_deviation_from_1": float(np.max(abs_dev)),
        "fraction_within_1pct": float(np.mean(abs_dev <= 0.01)),
        "fraction_within_5pct": float(np.mean(abs_dev <= 0.05)),
        "fraction_within_10pct": float(np.mean(abs_dev <= 0.10)),
    }

    # -------------------------------------------------------------------------
    # 2) Current nuisance + explicit q_pi
    # -------------------------------------------------------------------------
    t0 = time.time()
    cur = audit_crossfit(
        X, action, reward, pscore, position, action_context,
        pi0, pi1, seed, "current_sgd",
    )
    current_seconds = time.time() - t0

    q0_diff = cur["q0_explicit"] - cur["q0_short"]
    q1_diff = cur["q1_explicit"] - cur["q1_short"]
    qpi_summary = {
        "campaign": campaign,
        "direction": direction,
        "n": n,
        "current_model_seconds": current_seconds,
        "q0_mean_abs_explicit_minus_shortcut": float(np.mean(np.abs(q0_diff))),
        "q0_p99_abs_explicit_minus_shortcut": float(np.quantile(np.abs(q0_diff), 0.99)),
        "q0_max_abs_explicit_minus_shortcut": float(np.max(np.abs(q0_diff))),
        "q1_mean_abs_explicit_minus_shortcut": float(np.mean(np.abs(q1_diff))),
        "q1_p99_abs_explicit_minus_shortcut": float(np.quantile(np.abs(q1_diff), 0.99)),
        "q1_max_abs_explicit_minus_shortcut": float(np.max(np.abs(q1_diff))),
    }

    # -------------------------------------------------------------------------
    # 3) Symmetric w0/w1 diagnostics
    # -------------------------------------------------------------------------
    wrows = []
    for name, w in [("w0_baseline", cur["w0"]), ("w1_candidate", cur["w1"])]:
        w = np.asarray(w, dtype=float)
        wrows.append({
            "campaign": campaign,
            "direction": direction,
            "weight": name,
            "n": n,
            "ess": ess(w),
            "ess_fraction": ess(w) / max(n, 1),
            **qstats(w, "w_"),
            "fraction_above_clip": float(np.mean(w > WEIGHT_CLIP)),
        })

    # -------------------------------------------------------------------------
    # 4) Estimator sensitivities
    # -------------------------------------------------------------------------
    est = estimator_rows(cur, group, reward, "current")

    logistic_seconds = np.nan
    if RUN_LOGISTIC_NUISANCE:
        t1 = time.time()
        logit = audit_crossfit(
            X, action, reward, pscore, position, action_context,
            pi0, pi1, seed + 77_777, "logistic_sgd",
        )
        logistic_seconds = time.time() - t1
        est_logit = estimator_rows(logit, group, reward, "logistic")
        # IPS/SNIPS do not depend on nuisance; keep only logistic DM/DR rows.
        est_logit = est_logit[est_logit["estimator"].str.startswith("logistic_")]
        est = pd.concat([est, est_logit], ignore_index=True)

    compared = compare_to_reference(est, reference, campaign, direction)

    # -------------------------------------------------------------------------
    # Saved-result reproduction check on the exact 50k/rep0 sample.
    # -------------------------------------------------------------------------
    saved = saved_points[
        saved_points["campaign"].eq(campaign)
        & saved_points["direction"].eq(direction)
        & pd.to_numeric(saved_points["n"], errors="coerce").eq(n)
        & pd.to_numeric(saved_points["rep"], errors="coerce").eq(AUDIT_REP)
        & saved_points["estimator"].isin(["dm", "ips", "snips", "dr"])
    ][["group", "estimator", "estimate"]].copy()
    saved = saved.rename(columns={"estimate": "saved_estimate"})

    map_current = {
        "dm": "current_dm_shortcut",
        "ips": "ips_raw",
        "snips": "snips_raw",
        "dr": "current_dr_shortcut_raw",
    }
    rep_rows = []
    for saved_name, audit_name in map_current.items():
        a = est[est["estimator"].eq(audit_name)][["group", "estimate"]].copy()
        a = a.rename(columns={"estimate": "audit_estimate"})
        b = saved[saved["estimator"].eq(saved_name)][["group", "saved_estimate"]]
        z = a.merge(b, on="group", how="inner")
        if z.empty:
            continue
        diff = z["audit_estimate"] - z["saved_estimate"]
        rep_rows.append({
            "campaign": campaign,
            "direction": direction,
            "estimator": saved_name,
            "n_compared_groups": int(len(z)),
            "max_abs_difference": float(np.max(np.abs(diff))),
            "mean_abs_difference": float(np.mean(np.abs(diff))),
            "pass_1e_6": bool(np.max(np.abs(diff)) <= 1e-6),
        })

    return {
        "identity": pd.DataFrame([identity]),
        "qpi": pd.DataFrame([qpi_summary]),
        "weights": pd.DataFrame(wrows),
        "comparison": compared,
        "reproduction": pd.DataFrame(rep_rows),
        "current_seconds": current_seconds,
        "logistic_seconds": logistic_seconds,
    }


# =============================================================================
# Summaries, plots, conclusion
# =============================================================================

def summarize_comparison(comp: pd.DataFrame) -> pd.DataFrame:
    clear = comp[
        comp["reference_status"].isin(["beneficial", "harmful"])
        & (pd.to_numeric(comp["group"], errors="coerce") >= 0)
    ].copy()
    if clear.empty:
        return pd.DataFrame()

    rows = []
    for (campaign, direction, estimator), d in clear.groupby(
        ["campaign", "direction", "estimator"]
    ):
        pop = pd.to_numeric(d["population_weight"], errors="coerce").to_numpy(float)
        pop = pop / max(np.sum(pop), EPS)
        ae = pd.to_numeric(d["abs_error_vs_reference"], errors="coerce").to_numpy(float)
        er = pd.to_numeric(d["error_vs_reference"], errors="coerce").to_numpy(float)
        sg = pd.to_numeric(d["sign_agreement"], errors="coerce").to_numpy(float)
        rows.append({
            "campaign": campaign,
            "direction": direction,
            "estimator": estimator,
            "n_clear_groups": int(len(d)),
            "mae_unweighted": float(np.mean(ae)),
            "mae_population_weighted": float(np.sum(pop * ae)),
            "signed_bias_unweighted": float(np.mean(er)),
            "sign_agreement_unweighted": float(np.mean(sg)),
        })
    return pd.DataFrame(rows).sort_values(
        ["campaign", "direction", "mae_population_weighted"]
    )


def make_plots(identity: pd.DataFrame, summary: pd.DataFrame) -> None:
    if not summary.empty:
        keep = [
            "ips_raw",
            "snips_raw",
            "current_dr_shortcut_raw",
            "current_dr_explicit_raw",
            f"current_dr_shortcut_clip{WEIGHT_CLIP:g}",
            f"current_dr_explicit_clip{WEIGHT_CLIP:g}",
            "logistic_dr_explicit_raw",
            f"logistic_dr_explicit_clip{WEIGHT_CLIP:g}",
        ]
        d = summary[summary["estimator"].isin(keep)].copy()
        if len(d):
            labels = []
            vals = []
            for r in d.itertuples():
                labels.append(
                    ("R->BTS" if r.direction == "random_to_bts" else "BTS->R")
                    + " | " + r.estimator
                )
                vals.append(r.mae_population_weighted)
            y = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(labels))))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Population-weighted MAE vs independent on-policy reference")
            ax.set_title("OBD DR audit: which change explains the reference disagreement?")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            p = OUT_DIR / "fig_audit_estimator_mae.png"
            fig.savefig(p, dpi=260, bbox_inches="tight")
            plt.close(fig)
            print(f"[figure] {p}")

    if not identity.empty:
        d = identity.copy()
        labels = [
            ("R->BTS" if x == "random_to_bts" else "BTS->R")
            for x in d["direction"]
        ]
        vals = d["p95_abs_deviation_from_1"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(labels, vals)
        ax.axhline(0.05, linestyle="--", linewidth=1.2, label="5% deviation")
        ax.set_ylabel("95th percentile of |pi0/pscore - 1|")
        ax.set_title("Behavior-policy identity audit")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        p = OUT_DIR / "fig_audit_behavior_identity.png"
        fig.savefig(p, dpi=260, bbox_inches="tight")
        plt.close(fig)
        print(f"[figure] {p}")


def build_conclusion(
    identity: pd.DataFrame,
    qpi: pd.DataFrame,
    weights: pd.DataFrame,
    summary: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    rows = []

    # Reproduction first: if audit cannot reproduce the saved sample, stop causal interpretation.
    rep_ok = bool(len(reproduction) and reproduction["pass_1e_6"].all())
    rows.append({
        "diagnostic": "saved_50k_result_reproduction",
        "status": "PASS" if rep_ok else "FAIL",
        "evidence": (
            f"max abs difference={reproduction['max_abs_difference'].max():.3g}"
            if len(reproduction) else "no matching saved 50k/rep0 rows"
        ),
        "interpretation": (
            "The audit reproduces the original paper-run sample and can be compared causally."
            if rep_ok else
            "Do not interpret later diagnostics until source/config/sample reproduction is resolved."
        ),
    })

    # Behavior identity.
    for direction in ["random_to_bts", "bts_to_random"]:
        d = identity[identity["direction"].eq(direction)]
        if d.empty:
            continue
        r = d.iloc[0]
        p95 = float(r["p95_abs_deviation_from_1"])
        frac5 = float(r["fraction_within_5pct"])
        ok = (p95 <= 0.05) and (frac5 >= 0.95)
        rows.append({
            "diagnostic": f"behavior_identity_{direction}",
            "status": "PASS" if ok else "RED_FLAG",
            "evidence": f"p95 |rho0-1|={p95:.4f}; fraction within 5%={frac5:.3f}",
            "interpretation": (
                "Reconstructed pi0 is broadly consistent with logged pscore."
                if ok else
                "Reconstructed pi0 is not sufficiently identical to the logged behavior propensity; "
                "this can invalidate the DR baseline side and must be fixed before paper claims."
            ),
        })

    # q_pi shortcut.
    for direction in ["random_to_bts", "bts_to_random"]:
        d = qpi[qpi["direction"].eq(direction)]
        if d.empty:
            continue
        r = d.iloc[0]
        mdiff = max(
            float(r["q0_mean_abs_explicit_minus_shortcut"]),
            float(r["q1_mean_abs_explicit_minus_shortcut"]),
        )
        status = "MATERIAL" if mdiff > 5e-4 else ("SMALL" if mdiff > 1e-5 else "NEGLIGIBLE")
        rows.append({
            "diagnostic": f"qpi_shortcut_{direction}",
            "status": status,
            "evidence": f"max mean |explicit-shortcut|={mdiff:.6g}",
            "interpretation": (
                "Post-clipping expected-action shortcut materially changes q_pi."
                if status == "MATERIAL" else
                "Expected-action shortcut is unlikely to be the main source of the large OBD DR discrepancy."
            ),
        })

    # ESS / weight stress.
    for direction in ["random_to_bts", "bts_to_random"]:
        d = weights[
            weights["direction"].eq(direction)
            & weights["weight"].eq("w1_candidate")
        ]
        if d.empty:
            continue
        r = d.iloc[0]
        ef = float(r["ess_fraction"])
        rows.append({
            "diagnostic": f"candidate_weight_support_{direction}",
            "status": "SEVERE" if ef < 0.05 else ("WEAK" if ef < 0.10 else "OK"),
            "evidence": f"ESS/n={ef:.4f}; w99={float(r['w_q99']):.3g}; wmax={float(r['w_max']):.3g}",
            "interpretation": (
                "Candidate OPE is below the pre-specified 5% ESS ablation threshold."
                if ef < 0.05 else
                "Candidate support is not below the 5% safeguard threshold."
            ),
        })

    # Which estimator fix helps? Use population-weighted MAE in each direction.
    for direction in ["random_to_bts", "bts_to_random"]:
        d = summary[summary["direction"].eq(direction)].set_index("estimator")
        if d.empty or "current_dr_shortcut_raw" not in d.index:
            continue
        base = float(d.loc["current_dr_shortcut_raw", "mae_population_weighted"])
        candidates = {}
        for name in [
            "current_dr_explicit_raw",
            f"current_dr_shortcut_clip{WEIGHT_CLIP:g}",
            f"current_dr_explicit_clip{WEIGHT_CLIP:g}",
            "logistic_dr_explicit_raw",
            f"logistic_dr_explicit_clip{WEIGHT_CLIP:g}",
            "ips_raw",
            "snips_raw",
        ]:
            if name in d.index:
                candidates[name] = float(d.loc[name, "mae_population_weighted"])
        best_name = min(candidates, key=candidates.get) if candidates else "none"
        best = candidates.get(best_name, np.nan)
        rows.append({
            "diagnostic": f"estimator_sensitivity_{direction}",
            "status": "IMPROVEMENT_FOUND" if np.isfinite(best) and best < 0.5 * base else "NO_SINGLE_FIX",
            "evidence": f"current DR MAE={base:.6g}; best={best_name} MAE={best:.6g}",
            "interpretation": (
                "At least one targeted change cuts DR/reference error by >50%; this identifies a concrete mechanism to fix/test."
                if np.isfinite(best) and best < 0.5 * base else
                "No single tested change explains most of the DR/reference gap; keep OBD DR claims limited."
            ),
        })

    summary_rows = pd.DataFrame(rows)

    red_behavior = summary_rows[
        summary_rows["diagnostic"].str.startswith("behavior_identity_")
        & summary_rows["status"].eq("RED_FLAG")
    ]
    material_q = summary_rows[
        summary_rows["diagnostic"].str.startswith("qpi_shortcut_")
        & summary_rows["status"].eq("MATERIAL")
    ]
    severe_support = summary_rows[
        summary_rows["diagnostic"].str.startswith("candidate_weight_support_")
        & summary_rows["status"].eq("SEVERE")
    ]

    if not rep_ok:
        headline = (
            "STOP: the audit did not reproduce the saved 50k/rep0 OBD result closely enough. "
            "First resolve source/config/version mismatch. Do not rerun any large experiment yet."
        )
    elif len(red_behavior):
        headline = (
            "PRIMARY FINDING: reconstructed behavior-policy probabilities do not match logged pscore closely enough. "
            "Fix/clarify BTS policy reconstruction first, then rerun ONLY the OBD validation layer."
        )
    elif len(material_q):
        headline = (
            "PRIMARY FINDING: the q_pi expected-action-context shortcut is materially altered by clipping. "
            "Replace it with explicit per-action policy averaging, then rerun ONLY OBD."
        )
    elif len(severe_support):
        headline = (
            "PRIMARY FINDING: the remaining OBD failure is dominated by severe support/evaluability collapse. "
            "Do not rerun Synthetic or Semi. Treat ESS safeguard/stabilized DR as a pre-specified ablation and "
            "limit raw reverse-DR claims."
        )
    else:
        headline = (
            "No single implementation defect is identified. The OBD DR discrepancy should be reported as a real-log "
            "limitation; no broad new experiment is justified."
        )

    text = (
        "BC-SPI TARGETED OBD DR AUDIT CONCLUSION\n"
        "=======================================\n\n"
        + headline
        + "\n\n"
        "This audit is diagnostic only. It does not change the completed Synthetic or Semi-Synthetic evidence.\n"
        "If an OBD rerun is needed, rerun only the OBD layer after the specific issue above is corrected.\n"
    )
    return summary_rows, text


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("BC-SPI TARGETED OBD DR AUDIT")
    print("=" * 88)
    print(f"Python          : {sys.executable}")
    print(f"PROJECT_DIR     : {PROJECT_DIR}")
    print(f"OBD_DATA_PATH   : {cfg.OBD_DATA_PATH}")
    print(f"AUDIT_CAMPAIGNS : {AUDIT_CAMPAIGNS}")
    print(f"AUDIT_N         : {AUDIT_N:,} per direction")
    print(f"WEIGHT_CLIP     : {WEIGHT_CLIP:g} (pre-specified SWITCH_TAU)")
    print(f"OUTPUT_DIR      : {OUT_DIR}")

    require_file(TABLES_DIR / "obd_onpolicy_reference.csv")
    require_file(TABLES_DIR / "obd_point_estimates_long.csv")
    require_file(PROJECT_DIR / "run_obdnew.py")
    require_file(PROJECT_DIR / "bcspi_core.py")

    mock_self_test()

    if os.environ.get("BCSPI_OBD_AUDIT_MOCK_ONLY", "0") == "1":
        print("BCSPI_OBD_AUDIT_MOCK_ONLY=1 -> self-test only; external OBD skipped.")
        return

    if cfg.OBD_DATA_PATH is None or not Path(cfg.OBD_DATA_PATH).exists():
        raise FileNotFoundError(
            "Full OBD data path is missing. Set bcspi_config.OBD_DATA_PATH to the root "
            "containing random/all and bts/all before running this audit."
        )

    runner_helpers = load_obd_runner_helpers()
    load_pair = runner_helpers._load_pair

    reference = pd.read_csv(TABLES_DIR / "obd_onpolicy_reference.csv", low_memory=False)
    saved_points = pd.read_csv(TABLES_DIR / "obd_point_estimates_long.csv", low_memory=False)

    all_identity = []
    all_qpi = []
    all_weights = []
    all_comp = []
    all_rep = []
    timing = []

    for campaign in AUDIT_CAMPAIGNS:
        if campaign not in list(cfg.OBD_CAMPAIGNS):
            raise ValueError(f"Unknown campaign {campaign!r}; cfg.OBD_CAMPAIGNS={cfg.OBD_CAMPAIGNS}")
        ci = list(cfg.OBD_CAMPAIGNS).index(campaign)

        print(f"\n[LOAD] Full OBD campaign={campaign} ...")
        dsr, fr, _tr, dsb, fb, _tb, random_pos, bts_pos = load_pair(
            campaign, cfg.OBD_DATA_PATH
        )
        if int(fr["n_actions"]) != int(fb["n_actions"]):
            raise ValueError("Random/BTS action count mismatch.")
        if int(dsr.len_list) != int(dsb.len_list):
            raise ValueError("Random/BTS len_list mismatch.")
        L = int(dsr.len_list)

        print("[GROUP] fitting the exact frozen reward-blind group model once...")
        group_scaler, group_km = fit_group_model_exact(
            np.asarray(fr["context"], dtype=np.float32),
            int(cfg.RANDOM_SEED) + ci,
        )

        for direction in ["random_to_bts", "bts_to_random"]:
            res = run_direction(
                campaign, ci, direction,
                fr, fb, random_pos, bts_pos,
                group_scaler, group_km, L,
                reference, saved_points,
            )
            all_identity.append(res["identity"])
            all_qpi.append(res["qpi"])
            all_weights.append(res["weights"])
            all_comp.append(res["comparison"])
            all_rep.append(res["reproduction"])
            timing.append({
                "campaign": campaign,
                "direction": direction,
                "current_model_seconds": res["current_seconds"],
                "logistic_model_seconds": res["logistic_seconds"],
            })

        # Release the very large campaign objects before another campaign.
        del dsr, dsb, fr, fb, random_pos, bts_pos, group_scaler, group_km

    identity = pd.concat(all_identity, ignore_index=True)
    qpi = pd.concat(all_qpi, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    comparison = pd.concat(all_comp, ignore_index=True)
    reproduction = pd.concat(all_rep, ignore_index=True) if all_rep else pd.DataFrame()
    timing_df = pd.DataFrame(timing)

    save_csv(identity, "audit_behavior_identity.csv")
    save_csv(qpi, "audit_qpi_shortcut.csv")
    save_csv(weights, "audit_weight_diagnostics.csv")
    save_csv(comparison, "audit_estimator_reference_comparison.csv")
    save_csv(reproduction, "audit_saved_result_reproduction.csv")
    save_csv(timing_df, "audit_timing.csv")

    summary = summarize_comparison(comparison)
    save_csv(summary, "audit_estimator_reference_summary.csv")

    audit_summary, conclusion = build_conclusion(
        identity, qpi, weights, summary, reproduction
    )
    save_csv(audit_summary, "audit_summary.csv")

    p = OUT_DIR / "audit_conclusion.txt"
    p.write_text(conclusion, encoding="utf-8")
    print(f"[saved] {p}")

    make_plots(identity, summary)

    manifest = {
        "revision": "BCSPI_OBD_DR_TARGETED_AUDIT_R1",
        "project_dir": str(PROJECT_DIR),
        "obd_data_path": str(cfg.OBD_DATA_PATH),
        "campaigns": AUDIT_CAMPAIGNS,
        "audit_n_per_direction": AUDIT_N,
        "audit_rep": AUDIT_REP,
        "n_folds": AUDIT_N_FOLDS,
        "train_epochs": AUDIT_TRAIN_EPOCHS,
        "weight_clip": WEIGHT_CLIP,
        "run_logistic_nuisance": RUN_LOGISTIC_NUISANCE,
        "reuses_saved_reference": True,
        "reruns_full_obd_grid": False,
        "reruns_synthetic": False,
        "reruns_semi_synthetic": False,
    }
    save_json(manifest, "audit_manifest.json")

    print("\n" + conclusion)
    print("Audit outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
