"""BC-SPI Open Bandit Dataset (OBD) real-log validation.

Primary protocol
----------------
Random log -> evaluate production BernoulliTS, with an independent BTS on-policy
log as reference. Reverse BTS -> Random is the negative control. Deployment groups
are reward-blind user-context clusters crossed with position. Repeated behavior-log
subsampling measures evidence accumulation.

Important interpretation rules
------------------------------
1. OBD does NOT provide counterfactual truth. Independent on-policy logs are noisy
   references, not an oracle.
2. Local reference groups are classified as beneficial / harmful only when a robust
   two-sample binomial confidence interval for the CTR difference excludes zero.
   Otherwise they are "ambiguous" and are never counted as proven safe or harmful.
3. Therefore clear harmful exposure and ambiguous deployment exposure are reported
   separately. In particular, clear_harmful_exposure == 0 is NOT evidence of safety
   when most/all reference mass is ambiguous.
4. H8 is scored only when the independent reference has at least one clear local
   group. Otherwise its status is "pending_reference_precision".
5. Random and BTS raw user features are jointly encoded into one reward-blind feature
   basis before clustering, avoiding the 20-vs-22 dummy-column mismatch seen in the
   bundled 10k OBD logs.
6. OBD item-feature hashes can differ between Random and BTS even when the same item_id
   has the same underlying categorical profile.  We therefore verify equality of the
   item-feature partition structure up to a one-to-one category relabeling and build one
   canonical action_context indexed by item_id for BOTH logs.

Spyder instructions
-------------------
- Smoke mode: if obp is installed, OBD_DATA_PATH=None uses OBP's bundled 10k data.
- Paper mode: set bcspi_config.OBD_DATA_PATH to the FULL OBD root containing
  random/all, random/men, ..., bts/women.
- Run this file (F5). The fixed project/output paths come from bcspi_config.py;
  results are saved under C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code\bcspi_results.

Runner revision: PRE-PAPER-FINAL-R6-DRFIX. The runner prints and records this revision and
verifies the required revised output files before reporting completion.

The code performs an internal mock self-test before external OBD loading so that the
core OBD calculations and the revised reference/ambiguity accounting are validated
before a real-log run starts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import bcspi_config as cfg
from bcspi_core import (
    OBDComponents,
    crossfit_obd_components,
    dr_uncertainty_by_group,
    summaries_from_draw_dict,
    effective_sample_size,
    save_csv,
    save_json,
    save_show,
    print_frame,
    recency_weights,
)


EPS = 1e-12
RUNNER_REVISION = "PRE-PAPER-FINAL-R6-DRFIX"
REFERENCE_INTERVAL_METHOD = "newcombe_wilson_difference"
OBD_METRICS_VERSION = "obd_eval_v6_drfixed"

# Compute-efficient PAPER protocol: preserve all campaigns, both directions, all
# sample-size points, full-data evaluation, and 500/500 uncertainty draws, while
# reducing repeated non-full subsampling from the config default of 50 to 20.
PAPER_REPS_PER_NONFULL_SIZE = 20


# ============================================================================
# Revision / output self-identification
# ============================================================================

def _revision_sentinel_paths() -> list[Path]:
    """Files that uniquely identify the PRE-PAPER-FINAL-R6-DRFIX evaluation layer.

    They are removed at the beginning of a real run so stale outputs from an older
    run_obd.py cannot be mistaken for newly generated R4 results after a failed run.
    """
    return [
        cfg.TABLES_DIR / "obd_reference_precision_summary.csv",
        cfg.TABLES_DIR / "obd_evaluability_summary.csv",
        cfg.TABLES_DIR / "obd_h8_evidence_status.csv",
        cfg.RESULTS_DIR / "obd_run_manifest.json",
    ]


def _clear_revision_sentinels() -> None:
    for path in _revision_sentinel_paths():
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            raise RuntimeError(
                f"Could not remove stale OBD revision sentinel: {path}"
            ) from e


def _all_obd_output_paths() -> list[Path]:
    """Known OBD outputs produced by this runner.

    Only OBD files are listed here; Synthetic and Semi-Synthetic outputs are never
    touched. Clearing these at the start prevents stale CSVs from an older runner
    version from being mistaken for fresh results after a failed run.
    """
    return [
        cfg.TABLES_DIR / "obd_onpolicy_reference.csv",
        cfg.TABLES_DIR / "obd_point_estimates_long.csv",
        cfg.TABLES_DIR / "obd_gate_results_long.csv",
        cfg.TABLES_DIR / "obd_decision_results.csv",
        cfg.TABLES_DIR / "obd_safety_coverage_frontiers.csv",
        cfg.TABLES_DIR / "obd_evidence_diagnostics.csv",
        cfg.TABLES_DIR / "obd_temporal_robustness.csv",
        cfg.TABLES_DIR / "obd_reference_precision_summary.csv",
        cfg.TABLES_DIR / "obd_evaluability_summary.csv",
        cfg.TABLES_DIR / "obd_decision_summary.csv",
        cfg.TABLES_DIR / "obd_h8_evidence_status.csv",
        cfg.RESULTS_DIR / "obd_run_manifest.json",
    ]


def _clear_previous_obd_outputs() -> None:
    """Remove only stale OBD result files before a new run starts."""
    for path in _all_obd_output_paths():
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            raise RuntimeError(f"Could not remove stale OBD output: {path}") from e


def _verify_required_outputs() -> None:
    """Fail loudly unless the R3 interpretation-safe outputs were freshly produced."""
    csv_specs = {
        cfg.TABLES_DIR / "obd_reference_precision_summary.csv": {
            "campaign",
            "direction",
            "n_clear_groups",
            "n_ambiguous_groups",
            "clear_reference_weight",
            "ambiguous_reference_weight",
            "h8_reference_testability",
        },
        cfg.TABLES_DIR / "obd_evaluability_summary.csv": {
            "campaign",
            "direction",
            "mean_local_ess_fraction",
            "median_local_w_q99",
            "max_local_w_max",
            "min_local_pscore",
        },
        cfg.TABLES_DIR / "obd_h8_evidence_status.csv": {
            "campaign",
            "direction",
            "h8_state",
            "n_clear_groups",
            "n_ambiguous_groups",
            "ambiguous_reference_weight",
        },
        cfg.TABLES_DIR / "obd_decision_summary.csv": {
            "deployment_coverage",
            "clear_harmful_exposure",
            "ambiguous_deployment_exposure",
            "clear_beneficial_deployment_exposure",
            "reference_clear_policy_value",
            "reference_sign_selective_value",
        },
        cfg.TABLES_DIR / "obd_decision_results.csv": {
            "deployment_coverage",
            "clear_harmful_exposure",
            "ambiguous_deployment_exposure",
            "reference_clear_policy_value",
            "reference_sign_selective_value",
            "metrics_version",
            "runner_revision",
        },
        cfg.TABLES_DIR / "obd_safety_coverage_frontiers.csv": {
            "deployment_coverage",
            "clear_harmful_exposure",
            "ambiguous_deployment_exposure",
            "clear_beneficial_deployment_exposure",
            "clear_reference_weight",
            "ambiguous_reference_weight",
        },
    }

    for path, required_cols in csv_specs.items():
        if not path.exists():
            raise RuntimeError(
                f"PRE-PAPER-FINAL-R6-DRFIX output verification failed: missing file {path}"
            )
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(
                f"PRE-PAPER-FINAL-R6-DRFIX output verification failed while reading {path}"
            ) from e
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise RuntimeError(
                "PRE-PAPER-FINAL-R6-DRFIX output verification failed: "
                f"{path.name} is missing columns {missing}"
            )
        if df.empty:
            raise RuntimeError(
                f"PRE-PAPER-FINAL-R6-DRFIX output verification failed: {path.name} is empty."
            )
        forbidden = {"reference_oracle_value", "reference_oracle_regret", "oracle_regret"}
        bad = sorted(forbidden & set(df.columns))
        if bad:
            raise RuntimeError(
                "PRE-PAPER-FINAL-R6-DRFIX output verification failed: obsolete oracle "
                f"columns remain in {path.name}: {bad}"
            )

    manifest_path = cfg.RESULTS_DIR / "obd_run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"PRE-PAPER-FINAL-R6-DRFIX output verification failed: missing {manifest_path}"
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise RuntimeError(
            "PRE-PAPER-FINAL-R6-DRFIX output verification failed while reading manifest."
        ) from e

    expected = {
        "runner_revision": RUNNER_REVISION,
        "metrics_version": OBD_METRICS_VERSION,
        "action_context_alignment": "item_id_plus_bijective_categorical_relabeling",
        "obd_reward_nuisance": "crossfit_logistic_sgd",
        "q_pi_computation": "explicit_per_action_probability_average",
        "behavior_baseline": "logged_on_policy_w0_equals_1",
        "oracle_metrics_present": False,
        "ambiguity_aware_reporting": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                "PRE-PAPER-FINAL-R6-DRFIX manifest verification failed: "
                f"{key}={manifest.get(key)!r}, expected {value!r}."
            )

    print(
        "[PASS] PRE-PAPER-FINAL-R6-DRFIX output self-check: revised summaries, ambiguity "
        "accounting, no-oracle semantics, and manifest revision are present."
    )


# ============================================================================
# External package / data loading
# ============================================================================

def _import_obp():
    try:
        from obp.dataset import OpenBanditDataset
        from obp.policy import BernoulliTS
    except Exception as e:
        raise RuntimeError(
            "Open Bandit Pipeline is required for the real OBD run. Install the "
            "current 'obp' package in the SAME Python environment used by Spyder. "
            "The rest of the project does not depend on OBP. Original import error: "
            + repr(e)
        ) from e
    return OpenBanditDataset, BernoulliTS


def _validate_policy_matrix(name: str, p: np.ndarray, n_positions: int, n_actions: int) -> None:
    p = np.asarray(p, dtype=float)
    if p.shape != (n_positions, n_actions):
        raise ValueError(
            f"{name}: expected shape {(n_positions, n_actions)}, got {p.shape}."
        )
    if not np.isfinite(p).all():
        raise ValueError(f"{name}: contains NaN/Inf probabilities.")
    if np.any(p < -1e-12):
        raise ValueError(f"{name}: contains negative probabilities.")
    row_err = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
    if row_err > 1e-6:
        raise ValueError(f"{name}: row sums deviate from one by {row_err:.3g}.")


def _policy_matrices(dataset, campaign, BernoulliTS):
    """Known evaluation-policy matrices used by the OBD paper protocol.

    OBP's BernoulliTS.compute_batch_action_dist() computes one Monte-Carlo
    distribution and tiles it along n_rounds, so n_rounds=1 is sufficient for
    this static target-policy matrix.

    R6 correction: this reconstructed BTS matrix is never used as the behavior
    denominator for BTS logged data. The logged pscore remains authoritative.
    """
    k = int(dataset.n_actions)
    L = int(dataset.len_list)
    random_by_pos = np.full((L, k), 1.0 / k, dtype=float)

    bts = BernoulliTS(
        n_actions=k,
        len_list=L,
        is_zozotown_prior=True,
        campaign=campaign,
        random_state=cfg.RANDOM_SEED,
    )
    dist = bts.compute_batch_action_dist(
        n_sim=cfg.OBD_BTS_N_SIM,
        n_rounds=1,
    )
    dist = np.asarray(dist, dtype=float)
    if dist.shape != (1, k, L):
        raise ValueError(
            f"Unexpected BernoulliTS action_dist shape {dist.shape}; "
            f"expected (1,{k},{L})."
        )

    bts_by_pos = dist[0].T.copy()  # (position, action)
    row_sum = bts_by_pos.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("BernoulliTS reconstruction produced a zero-probability row.")
    bts_by_pos /= row_sum

    _validate_policy_matrix("Random policy", random_by_pos, L, k)
    _validate_policy_matrix("BTS policy", bts_by_pos, L, k)
    return random_by_pos, bts_by_pos


def _build_aligned_user_contexts(ds_r, ds_b):
    """Build Random/BTS user contexts in one shared reward-blind feature basis.

    OBP preprocesses each policy log separately, so independent dummy encoding can
    yield different columns.  For paper-scale logs we avoid concatenating all rows:
    categorical columns first receive a COMMON category set, each log is encoded
    separately, and the encoded columns are then aligned.  Numeric user features are
    passed through unchanged.  No reward/click/pscore/action information is used.
    """
    if not (hasattr(ds_r, "data") and hasattr(ds_b, "data")):
        raise ValueError(
            "OpenBanditDataset objects must expose raw .data for aligned preprocessing."
        )

    cols_r = [c for c in ds_r.data.columns if "user_feature" in str(c)]
    cols_b = [c for c in ds_b.data.columns if "user_feature" in str(c)]
    if cols_r != cols_b:
        raise ValueError(
            "Random/BTS raw user-feature columns differ. "
            f"random={cols_r}, bts={cols_b}"
        )
    if not cols_r:
        raise ValueError("No user_feature columns found in OBD raw data.")

    # Separate working frames avoid the paper-scale memory spike from concatenating
    # both full logs before get_dummies().
    rw = ds_r.data.loc[:, cols_r].copy()
    bw = ds_b.data.loc[:, cols_r].copy()

    for col in cols_r:
        sr, sb = rw[col], bw[col]
        categorical = (
            pd.api.types.is_object_dtype(sr.dtype)
            or pd.api.types.is_object_dtype(sb.dtype)
            or pd.api.types.is_string_dtype(sr.dtype)
            or pd.api.types.is_string_dtype(sb.dtype)
            or isinstance(sr.dtype, pd.CategoricalDtype)
            or isinstance(sb.dtype, pd.CategoricalDtype)
        )
        if categorical:
            # Stable union: Random levels first, then unseen BTS levels. This fixes
            # the dropped-reference category as well as the final dummy columns.
            cats = list(pd.unique(sr.dropna()))
            for value in pd.unique(sb.dropna()):
                if value not in cats:
                    cats.append(value)
            dtype = pd.CategoricalDtype(categories=cats)
            rw[col] = pd.Categorical(sr, dtype=dtype)
            bw[col] = pd.Categorical(sb, dtype=dtype)

    er = pd.get_dummies(rw, drop_first=True)
    eb = pd.get_dummies(bw, drop_first=True)
    common_cols = er.columns.union(eb.columns, sort=False)
    er = er.reindex(columns=common_cols, fill_value=0)
    eb = eb.reindex(columns=common_cols, fill_value=0)

    Xr = er.to_numpy(dtype=np.float32)
    Xb = eb.to_numpy(dtype=np.float32)
    del rw, bw, er, eb

    if Xr.shape[1] != Xb.shape[1]:
        raise AssertionError(
            "Aligned OBD contexts unexpectedly have different dimensions."
        )
    if Xr.shape[1] == 0:
        raise ValueError("Aligned OBD user context has zero columns.")
    if not (np.isfinite(Xr).all() and np.isfinite(Xb).all()):
        raise ValueError("Aligned OBD context contains NaN/Inf values.")

    print(
        f"  aligned user context: random {Xr.shape}, bts {Xb.shape} "
        f"(shared {Xr.shape[1]} features)"
    )
    return Xr, Xb


def _categorical_partitions_equivalent(a: pd.Series, b: pd.Series) -> bool:
    """Return True when two categorical vectors differ only by a bijective relabeling.

    OBD anonymized item-feature hashes may be policy-log specific.  For action-context
    alignment we care about whether the SAME item_ids induce the same category
    partition, not whether the literal hashes are equal.
    """
    if len(a) != len(b):
        return False

    # Preserve missingness as a real level for the structural check.  OBD currently
    # has no missing item-feature values, but this makes the validation explicit.
    aa = a.astype("string").fillna("<BCSPI_NA>")
    bb = b.astype("string").fillna("<BCSPI_NA>")
    pairs = pd.DataFrame({"a": aa.to_numpy(), "b": bb.to_numpy()})

    # a -> b must be a function and b -> a must also be a function.  Together these
    # conditions mean the two labels encode exactly the same equivalence classes.
    a_to_b = pairs.groupby("a", dropna=False)["b"].nunique(dropna=False)
    b_to_a = pairs.groupby("b", dropna=False)["a"].nunique(dropna=False)
    return bool((a_to_b <= 1).all() and (b_to_a <= 1).all())


def _build_aligned_action_contexts(ds_r, ds_b, n_actions: int) -> np.ndarray:
    """Build one canonical item/action context for Random and BTS logs.

    Why this is necessary
    ---------------------
    OBP loads ``item_context.csv`` separately for Random and BTS and applies a
    ``LabelEncoder`` separately to the anonymized categorical item features.  The
    literal hashes can differ between the two policy logs even when item_id denotes
    the same arm and the underlying category partition is the same.  Comparing the
    two already-encoded ``action_context`` matrices elementwise is therefore too
    strict and caused the PRE-PAPER-FINAL-R4 smoke run to stop.

    Safety rule
    -----------
    We do NOT blindly take one matrix.  First we verify that:
      * both raw item tables cover exactly item_id = 0,...,n_actions-1;
      * their item-feature column names agree;
      * the numeric item_feature_0 is equal across policy logs; and
      * every categorical item feature induces the same partition of item_ids up to
        a one-to-one relabeling of the anonymized hashes.

    Only after those checks do we construct a canonical matrix, using the Random-log
    labels as the arbitrary but fixed coding.  The same matrix is then supplied to
    both behavior directions, so action_context[action] has identical semantics.
    """
    if not (hasattr(ds_r, "item_context") and hasattr(ds_b, "item_context")):
        raise ValueError(
            "OpenBanditDataset objects must expose raw .item_context for safe "
            "Random/BTS action-context alignment."
        )

    ir = ds_r.item_context.copy()
    ib = ds_b.item_context.copy()
    if "item_id" not in ir.columns or "item_id" not in ib.columns:
        raise ValueError("OBD item_context must contain an item_id column.")

    cols_r = [c for c in ir.columns if c != "item_id"]
    cols_b = [c for c in ib.columns if c != "item_id"]
    if cols_r != cols_b:
        raise ValueError(
            "Random/BTS raw item-feature columns differ. "
            f"random={cols_r}, bts={cols_b}"
        )
    if "item_feature_0" not in cols_r:
        raise ValueError(
            "Expected OBD item_feature_0 in raw item_context; cannot reproduce the "
            "official action-context layout safely."
        )

    def _indexed(df: pd.DataFrame, label: str) -> pd.DataFrame:
        ids_num = pd.to_numeric(df["item_id"], errors="coerce")
        if ids_num.isna().any():
            raise ValueError(f"{label}: non-numeric item_id detected in item_context.")
        ids = ids_num.to_numpy(dtype=int)
        if not np.allclose(ids_num.to_numpy(dtype=float), ids.astype(float)):
            raise ValueError(f"{label}: non-integer item_id detected in item_context.")
        if pd.Series(ids).duplicated().any():
            raise ValueError(f"{label}: duplicate item_id detected in item_context.")
        expected = np.arange(int(n_actions), dtype=int)
        if not np.array_equal(np.sort(ids), expected):
            raise ValueError(
                f"{label}: item_context item_id support does not equal "
                f"0..{int(n_actions)-1}."
            )
        out = df.copy()
        out["item_id"] = ids
        return out.set_index("item_id").loc[expected, cols_r]

    rr = _indexed(ir, "Random")
    bb = _indexed(ib, "BTS")

    # item_feature_0 is numeric in the public OBD preprocessing and is appended after
    # the LabelEncoded categorical columns.  It should be identical for the same arm.
    nr = pd.to_numeric(rr["item_feature_0"], errors="coerce").to_numpy(dtype=float)
    nb = pd.to_numeric(bb["item_feature_0"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(nr).all() and np.isfinite(nb).all()):
        raise ValueError("OBD item_feature_0 contains non-numeric/NaN values.")
    if not np.allclose(nr, nb, rtol=1e-10, atol=1e-12):
        max_diff = float(np.max(np.abs(nr - nb)))
        raise ValueError(
            "Random/BTS numeric item_feature_0 differs for the same item_id; "
            f"max absolute difference={max_diff:.6g}. Action identities are not "
            "safe to align automatically."
        )

    categorical_cols = [c for c in cols_r if c != "item_feature_0"]
    encoded_cols = []
    for col in categorical_cols:
        if not _categorical_partitions_equivalent(rr[col], bb[col]):
            raise ValueError(
                "Random/BTS categorical item feature does not match up to a "
                f"one-to-one relabeling for column {col!r}. Action identities are "
                "not safe to align automatically."
            )

        # Match the semantics of sklearn LabelEncoder: deterministic sorted labels.
        sr = rr[col].astype("string").fillna("<BCSPI_NA>")
        levels = sorted(pd.unique(sr).tolist())
        code_map = {value: j for j, value in enumerate(levels)}
        encoded_cols.append(sr.map(code_map).to_numpy(dtype=float))

    # Reproduce OBP's layout: categorical item features first, item_feature_0 last.
    pieces = encoded_cols + [nr.astype(float)]
    action_context = np.column_stack(pieces).astype(np.float32)

    if action_context.shape[0] != int(n_actions):
        raise AssertionError("Aligned OBD action_context row count is incorrect.")
    if action_context.ndim != 2 or action_context.shape[1] == 0:
        raise ValueError("Aligned OBD action_context has invalid shape.")
    if not np.isfinite(action_context).all():
        raise ValueError("Aligned OBD action_context contains NaN/Inf values.")

    print(
        f"  aligned action context: {action_context.shape} "
        f"(item_id aligned; {len(categorical_cols)} categorical features verified "
        "up to bijective hash relabeling)"
    )
    return action_context


def _validate_feedback(feedback: dict, name: str) -> None:
    required = [
        "n_rounds",
        "n_actions",
        "context",
        "action",
        "reward",
        "pscore",
        "position",
        "action_context",
    ]
    missing = [x for x in required if x not in feedback]
    if missing:
        raise ValueError(f"{name}: missing required feedback keys {missing}.")

    n = int(feedback["n_rounds"])
    k = int(feedback["n_actions"])
    X = np.asarray(feedback["context"])
    a = np.asarray(feedback["action"])
    r = np.asarray(feedback["reward"], dtype=float)
    p = np.asarray(feedback["pscore"], dtype=float)
    pos = np.asarray(feedback["position"])
    ac = np.asarray(feedback["action_context"])

    if n <= 1:
        raise ValueError(f"{name}: n_rounds must exceed 1.")
    if k <= 1:
        raise ValueError(f"{name}: n_actions must exceed 1.")
    for key, arr in [("context", X), ("action", a), ("reward", r), ("pscore", p), ("position", pos)]:
        if len(arr) != n:
            raise ValueError(f"{name}: {key} length {len(arr)} != n_rounds {n}.")
    if X.ndim != 2:
        raise ValueError(f"{name}: context must be 2D, got shape {X.shape}.")
    if ac.ndim != 2 or ac.shape[0] != k:
        raise ValueError(
            f"{name}: action_context must have n_actions={k} rows, got {ac.shape}."
        )
    if not (np.isfinite(X).all() and np.isfinite(r).all() and np.isfinite(p).all() and np.isfinite(ac).all()):
        raise ValueError(f"{name}: NaN/Inf detected in context/reward/pscore/action_context.")
    if np.any(p <= 0) or np.any(p > 1 + 1e-10):
        raise ValueError(f"{name}: pscore must lie in (0,1].")
    if np.any(a < 0) or np.any(a >= k):
        raise ValueError(f"{name}: action outside [0, n_actions).")
    if np.any(pos < 0):
        raise ValueError(f"{name}: negative position detected.")
    # OBD is a click/no-click dataset. The reference interval below is binomial.
    if not np.all(np.isclose(r, 0.0) | np.isclose(r, 1.0)):
        raise ValueError(
            f"{name}: OBD reward is expected to be binary click/no-click (0/1)."
        )


def _load_pair(campaign, data_path=None):
    OpenBanditDataset, BernoulliTS = _import_obp()
    kwargs = {} if data_path is None else {"data_path": Path(data_path)}

    ds_r = OpenBanditDataset(
        behavior_policy="random",
        campaign=campaign,
        **kwargs,
    )
    ds_b = OpenBanditDataset(
        behavior_policy="bts",
        campaign=campaign,
        **kwargs,
    )

    fr = ds_r.obtain_batch_bandit_feedback()
    fb = ds_b.obtain_batch_bandit_feedback()

    if int(fr["n_actions"]) != int(fb["n_actions"]):
        raise ValueError("Random/BTS n_actions mismatch.")

    # IMPORTANT: OBP preprocesses anonymized item features independently for the two
    # policy logs. Literal action_context values can therefore differ even when the
    # same item_id has the same underlying profile. Verify raw item-feature structure
    # up to policy-specific hash relabeling and then use ONE canonical action context.
    shared_action_context = _build_aligned_action_contexts(
        ds_r, ds_b, int(fr["n_actions"])
    )

    # Build one shared reward-blind user-feature representation before clustering
    # and nuisance fitting (fixes the independent dummy-basis problem).
    Xr_aligned, Xb_aligned = _build_aligned_user_contexts(ds_r, ds_b)
    fr = dict(fr)
    fb = dict(fb)
    fr["context"] = Xr_aligned
    fb["context"] = Xb_aligned
    fr["action_context"] = shared_action_context
    fb["action_context"] = shared_action_context

    _validate_feedback(fr, f"OBD random/{campaign}")
    _validate_feedback(fb, f"OBD bts/{campaign}")

    random_pos, bts_pos = _policy_matrices(ds_r, campaign, BernoulliTS)
    L = int(ds_r.len_list)
    if int(ds_b.len_list) != L:
        raise ValueError("Random/BTS len_list mismatch.")
    if np.max(np.asarray(fr["position"], dtype=int)) >= L or np.max(np.asarray(fb["position"], dtype=int)) >= L:
        raise ValueError("Observed OBD position exceeds dataset len_list.")

    # Raw OpenBanditDataset data are sorted by timestamp by the loader.
    tr = (
        np.asarray(ds_r.data["timestamp"]).astype(str)
        if hasattr(ds_r, "data") and "timestamp" in ds_r.data
        else None
    )
    tb = (
        np.asarray(ds_b.data["timestamp"]).astype(str)
        if hasattr(ds_b, "data") and "timestamp" in ds_b.data
        else None
    )
    return ds_r, fr, tr, ds_b, fb, tb, random_pos, bts_pos


# ============================================================================
# Frozen reward-blind deployment groups
# ============================================================================

def _fit_groups(
    X_behavior,
    X_reference,
    pos_behavior,
    pos_reference,
    seed,
):
    X_behavior = np.asarray(X_behavior, dtype=np.float32)
    X_reference = np.asarray(X_reference, dtype=np.float32)
    pos_behavior = np.asarray(pos_behavior, dtype=int)
    pos_reference = np.asarray(pos_reference, dtype=int)

    if X_behavior.ndim != 2 or X_reference.ndim != 2:
        raise ValueError("OBD grouping requires 2D context matrices.")
    if X_behavior.shape[1] != X_reference.shape[1]:
        raise ValueError(
            "OBD grouping context dimensions differ after alignment: "
            f"{X_behavior.shape[1]} vs {X_reference.shape[1]}."
        )
    if len(X_behavior) < cfg.OBD_CLUSTER_K:
        raise ValueError(
            f"Need at least {cfg.OBD_CLUSTER_K} behavior observations for clustering."
        )

    n_design = min(len(X_behavior), cfg.OBD_CLUSTER_DESIGN_MAX_N)
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(X_behavior), size=n_design, replace=False)
        if n_design < len(X_behavior)
        else np.arange(len(X_behavior))
    )

    scaler = StandardScaler()
    Xd = scaler.fit_transform(X_behavior[idx])
    km = MiniBatchKMeans(
        n_clusters=cfg.OBD_CLUSTER_K,
        random_state=seed,
        batch_size=4096,
        n_init=10,
    )
    km.fit(Xd)

    # Chunked transform/predict avoids allocating another full scaled copy of a
    # multi-million-row OBD context matrix during the paper run.
    def predict_chunks(X, batch_size=250_000):
        out = np.empty(len(X), dtype=int)
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            out[start:stop] = km.predict(scaler.transform(X[start:stop]))
        return out

    cb = predict_chunks(X_behavior)
    cr = predict_chunks(X_reference)
    L = max(int(np.max(pos_behavior)), int(np.max(pos_reference))) + 1
    gb = cb * L + pos_behavior
    gr = cr * L + pos_reference
    return gb.astype(int), gr.astype(int), scaler, km, L


# ============================================================================
# Independent on-policy reference: robust low-CTR intervals, no "oracle"
# ============================================================================

def _wilson_interval(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    successes = int(successes)
    if successes < 0 or successes > n:
        raise ValueError("Binomial successes must lie in [0,n].")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n + z2 / (4.0 * n * n)
        )
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def _newcombe_difference_ci(
    x1: int,
    n1: int,
    x0: int,
    n0: int,
    alpha: float,
) -> tuple[float, float]:
    """Newcombe-style CI for p1-p0 using Wilson component intervals.

    This is preferable to a plain Wald interval for the very low CTR and small local
    subgroup counts present in the 10k bundled OBD data.
    """
    if n1 <= 0 or n0 <= 0:
        return np.nan, np.nan

    p1 = x1 / n1
    p0 = x0 / n0
    l1, u1 = _wilson_interval(x1, n1, alpha)
    l0, u0 = _wilson_interval(x0, n0, alpha)
    d = p1 - p0

    lower = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def _reference_status(lower: float, upper: float) -> str:
    # Safety classification is relative to zero improvement. DELTA is separately
    # used by the deployment Gate and may later be set to a positive practical margin.
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "unavailable"
    if lower > 0.0:
        return "beneficial"
    if upper < 0.0:
        return "harmful"
    return "ambiguous"


def _one_reference_row(
    group: int,
    rr: np.ndarray,
    rb: np.ndarray,
    direction: str,
) -> dict:
    rr = np.asarray(rr, dtype=float)
    rb = np.asarray(rb, dtype=float)
    nr = len(rr)
    nb = len(rb)
    if nr <= 0 or nb <= 0:
        raise ValueError("Reference row requires observations from both on-policy logs.")

    xr = int(np.rint(rr.sum()))
    xb = int(np.rint(rb.sum()))
    vr = float(rr.mean())
    vb = float(rb.mean())

    if direction == "random_to_bts":
        v0, v1 = vr, vb
        x0, n0 = xr, nr
        x1, n1 = xb, nb
    elif direction == "bts_to_random":
        v0, v1 = vb, vr
        x0, n0 = xb, nb
        x1, n1 = xr, nr
    else:
        raise ValueError(f"Unknown OBD direction: {direction}")

    delta = float(v1 - v0)
    lo, hi = _newcombe_difference_ci(
        x1=x1,
        n1=n1,
        x0=x0,
        n0=n0,
        alpha=cfg.OBD_REFERENCE_ALPHA,
    )
    status = _reference_status(lo, hi)

    # Keep a descriptive Wald SE for diagnostics only; CI/status use Newcombe-Wilson.
    se_wald = math.sqrt(
        max(v1 * (1.0 - v1), 0.0) / n1
        + max(v0 * (1.0 - v0), 0.0) / n0
    )

    return dict(
        group=int(group),
        n_random=nr,
        n_bts=nb,
        clicks_random=xr,
        clicks_bts=xb,
        population_weight_raw=nr + nb,
        v_random=vr,
        v_bts=vb,
        v0_ref=v0,
        v1_ref=v1,
        delta_ref=delta,
        ref_se=float(se_wald),
        ref_ci_lower=float(lo),
        ref_ci_upper=float(hi),
        ref_ci_width=float(hi - lo),
        reference_status=status,
        reference_interval_method=REFERENCE_INTERVAL_METHOD,
    )


def _reference_table_from_rewards(
    reward_random,
    g_random,
    reward_bts,
    g_bts,
    direction,
):
    rr_all = np.asarray(reward_random, dtype=float)
    rb_all = np.asarray(reward_bts, dtype=float)
    gr = np.asarray(g_random, dtype=int)
    gb = np.asarray(g_bts, dtype=int)

    if len(rr_all) != len(gr) or len(rb_all) != len(gb):
        raise ValueError("Reference reward/group arrays do not align.")
    if not np.all(np.isclose(rr_all, 0.0) | np.isclose(rr_all, 1.0)):
        raise ValueError("Random reference reward must be binary 0/1.")
    if not np.all(np.isclose(rb_all, 0.0) | np.isclose(rb_all, 1.0)):
        raise ValueError("BTS reference reward must be binary 0/1.")

    rows = []
    groups = sorted(set(np.unique(gr)) | set(np.unique(gb)))
    for g in groups:
        rr = rr_all[gr == g]
        rb = rb_all[gb == g]
        # A local on-policy comparison is only defined when both logs contain the group.
        if len(rr) == 0 or len(rb) == 0:
            continue
        rows.append(_one_reference_row(int(g), rr, rb, direction))

    out = pd.DataFrame(rows)
    if len(out):
        local_total = float(out["population_weight_raw"].sum())
        out["population_weight"] = out["population_weight_raw"] / max(local_total, EPS)

    # Add a global independent on-policy reference row. It is kept separate from the
    # local population-weight normalization and therefore has population_weight=1.
    global_row = _one_reference_row(-1, rr_all, rb_all, direction)
    global_row["population_weight"] = 1.0
    out = pd.concat([out, pd.DataFrame([global_row])], ignore_index=True)
    return out


def _reference_table(f_random, g_random, f_bts, g_bts, direction):
    return _reference_table_from_rewards(
        np.asarray(f_random["reward"], dtype=float),
        g_random,
        np.asarray(f_bts["reward"], dtype=float),
        g_bts,
        direction,
    )


# ============================================================================
# OPE estimates, diagnostics, Gate/reference metrics
# ============================================================================

def _group_point_estimates(comp: OBDComponents, group, reward):
    group = np.asarray(group, dtype=int)
    reward = np.asarray(reward, dtype=float)
    rows = []

    for g in sorted(np.unique(group).tolist()) + [-1]:
        m = np.ones(len(group), dtype=bool) if g == -1 else group == g
        dm = comp.q_pi1[m] - comp.q_pi0[m]
        ips = comp.w1[m] * reward[m] - comp.w0[m] * reward[m]
        dr = comp.dr_delta[m]
        sw = (
            comp.q_pi1[m]
            + (comp.w1[m] <= cfg.SWITCH_TAU)
            * comp.w1[m]
            * (reward[m] - comp.q_obs[m])
        ) - (
            comp.q_pi0[m]
            + (comp.w0[m] <= cfg.SWITCH_TAU)
            * comp.w0[m]
            * (reward[m] - comp.q_obs[m])
        )
        a1 = comp.w1[m]
        a0 = comp.w0[m]
        sn = (
            np.sum(a1 * reward[m]) / max(np.sum(a1), EPS)
            - np.sum(a0 * reward[m]) / max(np.sum(a0), EPS)
        )

        for name, val in [
            ("dm", dm.mean()),
            ("ips", ips.mean()),
            ("snips", sn),
            ("dr", dr.mean()),
            ("switch_dr", sw.mean()),
        ]:
            rows.append(
                dict(
                    group=int(g),
                    estimator=name,
                    estimate=float(val),
                )
            )
    return pd.DataFrame(rows)


def _obd_diagnostics(comp, group, pscore, base_weight=None):
    group = np.asarray(group, dtype=int)
    pscore = np.asarray(pscore, dtype=float)
    r = (
        np.ones(len(group), dtype=float)
        if base_weight is None
        else np.asarray(base_weight, dtype=float)
    )
    if len(r) != len(group):
        raise ValueError("OBD diagnostic base_weight length mismatch.")
    if np.any(r < 0) or not np.isfinite(r).all():
        raise ValueError("OBD diagnostic base_weight must be finite and nonnegative.")

    rows = []
    for g in sorted(np.unique(group).tolist()) + [-1]:
        m = np.ones(len(group), dtype=bool) if g == -1 else group == g
        w = np.asarray(comp.w1[m], dtype=float)
        joint = r[m] * w
        rows.append(
            dict(
                group=int(g),
                n_group=int(m.sum()),
                group_fraction=float(m.mean()),
                ess_pi1=effective_sample_size(w),
                ess_fraction=effective_sample_size(w) / max(int(m.sum()), 1),
                weighted_ess=effective_sample_size(joint),
                weighted_ess_fraction=effective_sample_size(joint) / max(int(m.sum()), 1),
                w_q95=float(np.quantile(w, 0.95)),
                w_q99=float(np.quantile(w, 0.99)),
                w_max=float(np.max(w)),
                pscore_min=float(np.min(pscore[m])),
            )
        )
    return pd.DataFrame(rows)


def _reference_metrics(gate_df, ref):
    """Evaluate a Gate against an independent on-policy reference without an oracle.

    Only reference groups whose Newcombe-Wilson CI excludes zero are considered
    clear. Ambiguous groups are tracked separately and are never silently counted as
    safe or harmful.
    """
    r = ref[ref["group"] >= 0].set_index("group")
    g = gate_df[gate_df["group"] >= 0].set_index("group")
    common = sorted(set(r.index) & set(g.index))
    if not common:
        return dict(
            deployment_coverage=np.nan,
            clear_harmful_exposure=np.nan,
            ambiguous_deployment_exposure=np.nan,
            clear_beneficial_deployment_exposure=np.nan,
            clear_sign_agreement=np.nan,
            false_deploy_clear_harm=np.nan,
            false_abstain_clear_benefit=np.nan,
            reference_hybrid_value=np.nan,
            reference_clear_policy_value=np.nan,
            hybrid_minus_reference_clear_policy=np.nan,
            reference_sign_selective_value=np.nan,
            hybrid_minus_reference_sign_selective=np.nan,
            reference_baseline_value=np.nan,
            reference_candidate_value=np.nan,
            clear_reference_weight=np.nan,
            ambiguous_reference_weight=np.nan,
            n_clear_groups=0,
            n_ambiguous_groups=0,
        )

    r = r.loc[common]
    g = g.loc[common]
    p = r["population_weight"].to_numpy(dtype=float)
    p = p / max(float(p.sum()), EPS)
    d = g["gate"].to_numpy(dtype=int)
    delta = r["delta_ref"].to_numpy(dtype=float)
    status = r["reference_status"].astype(str).to_numpy()

    beneficial = status == "beneficial"
    harmful = status == "harmful"
    ambiguous = status == "ambiguous"
    clear = beneficial | harmful

    v0 = r["v0_ref"].to_numpy(dtype=float)
    v1 = r["v1_ref"].to_numpy(dtype=float)

    hybrid_value = float(np.sum(p * (d * v1 + (1 - d) * v0)))

    # "Reference clear policy": switch only where the independent on-policy CI is
    # clearly beneficial; remain baseline in harmful and ambiguous groups.
    clear_policy = beneficial.astype(int)
    clear_policy_value = float(
        np.sum(p * (clear_policy * v1 + (1 - clear_policy) * v0))
    )

    # Descriptive sign-selective benchmark based only on the noisy reference point
    # estimate. It is NOT an oracle and must not be used as counterfactual truth.
    sign_policy = (delta > 0.0).astype(int)
    sign_policy_value = float(
        np.sum(p * (sign_policy * v1 + (1 - sign_policy) * v0))
    )

    return dict(
        deployment_coverage=float(np.sum(p * d)),
        clear_harmful_exposure=float(np.sum(p * d * harmful)),
        ambiguous_deployment_exposure=float(np.sum(p * d * ambiguous)),
        clear_beneficial_deployment_exposure=float(np.sum(p * d * beneficial)),
        clear_sign_agreement=(
            float(np.mean(d[clear] == beneficial[clear])) if clear.any() else np.nan
        ),
        false_deploy_clear_harm=(
            float(np.mean(d[harmful] == 1)) if harmful.any() else np.nan
        ),
        false_abstain_clear_benefit=(
            float(np.mean(d[beneficial] == 0)) if beneficial.any() else np.nan
        ),
        reference_hybrid_value=hybrid_value,
        reference_clear_policy_value=clear_policy_value,
        hybrid_minus_reference_clear_policy=float(hybrid_value - clear_policy_value),
        reference_sign_selective_value=sign_policy_value,
        hybrid_minus_reference_sign_selective=float(hybrid_value - sign_policy_value),
        reference_baseline_value=float(np.sum(p * v0)),
        reference_candidate_value=float(np.sum(p * v1)),
        clear_reference_weight=float(np.sum(p * clear)),
        ambiguous_reference_weight=float(np.sum(p * ambiguous)),
        n_clear_groups=int(clear.sum()),
        n_ambiguous_groups=int(ambiguous.sum()),
    )


def _reference_frontier(draws, ref):
    r = ref[ref["group"] >= 0].set_index("group")
    groups = sorted(set(r.index) & {x for x in draws if x >= 0})
    if not groups:
        return pd.DataFrame()

    p = r.loc[groups, "population_weight"].to_numpy(dtype=float)
    p = p / max(float(p.sum()), EPS)
    status = r.loc[groups, "reference_status"].astype(str).to_numpy()
    harmful = status == "harmful"
    beneficial = status == "beneficial"
    ambiguous = status == "ambiguous"
    probs = np.array(
        [np.mean(np.asarray(draws[g], dtype=float) > cfg.DELTA) for g in groups],
        dtype=float,
    )

    rows = []
    for th in cfg.POSTERIOR_GATE_THRESHOLDS:
        d = (probs >= th).astype(int)
        clear_harm = float(np.sum(p * d * harmful))
        rows.append(
            dict(
                posterior_threshold=float(th),
                deployment_coverage=float(np.sum(p * d)),
                clear_harmful_exposure=clear_harm,
                ambiguous_deployment_exposure=float(np.sum(p * d * ambiguous)),
                clear_beneficial_deployment_exposure=float(np.sum(p * d * beneficial)),
                clear_reference_weight=float(np.sum(p * (harmful | beneficial))),
                ambiguous_reference_weight=float(np.sum(p * ambiguous)),
                # Backward-compatible plotting alias. Interpret as CLEAR harmful only.
                harmful_exposure=clear_harm,
            )
        )
    return pd.DataFrame(rows)


def _run_one_sample(
    feedback,
    group,
    action_context,
    pi0_pos,
    pi1_pos,
    idx,
    seed,
    base_weight=None,
    baseline_is_behavior=True,
):
    idx = np.asarray(idx, dtype=int)
    X = np.asarray(feedback["context"])[idx]
    a = np.asarray(feedback["action"])[idx].astype(int)
    r = np.asarray(feedback["reward"])[idx].astype(float)
    p = np.asarray(feedback["pscore"])[idx].astype(float)
    pos = np.asarray(feedback["position"])[idx].astype(int)
    gr = np.asarray(group, dtype=int)[idx]

    if len(idx) < 2:
        raise ValueError("OBD sample must contain at least two rows.")

    comp = crossfit_obd_components(
        X,
        a,
        r,
        p,
        pos,
        np.asarray(action_context),
        pi0_pos,
        pi1_pos,
        cfg.OBD_N_FOLDS,
        seed,
        baseline_is_behavior=baseline_is_behavior,
    )

    bw = (
        np.ones(len(idx), dtype=float)
        if base_weight is None
        else np.asarray(base_weight, dtype=float)
    )
    if len(bw) != len(idx):
        raise ValueError("OBD base_weight length does not match sampled rows.")
    if np.any(bw < 0) or not np.isfinite(bw).all() or float(bw.sum()) <= 0:
        raise ValueError("OBD base_weight must be finite, nonnegative, and nonzero.")

    bb = dr_uncertainty_by_group(
        comp.dr_delta,
        gr,
        cfg.OBD_BAYES_BOOTSTRAP_DRAWS,
        seed + 11,
        bw,
        bayesian=True,
    )
    boot = dr_uncertainty_by_group(
        comp.dr_delta,
        gr,
        cfg.OBD_BOOTSTRAP_DRAWS,
        seed + 17,
        bw,
        bayesian=False,
    )

    return (
        comp,
        gr,
        r,
        p,
        _group_point_estimates(comp, gr, r),
        _obd_diagnostics(comp, gr, p, bw),
        {"bb_dr": bb, "bootstrap_dr": boot},
    )


# ============================================================================
# Paper-facing summaries
# ============================================================================

def _reference_precision_summary(refs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (campaign, direction), d in refs.groupby(["campaign", "direction"], dropna=False):
        local = d[d["group"] >= 0].copy()
        glob = d[d["group"] == -1].copy()
        if local.empty:
            continue

        p = local["population_weight"].to_numpy(dtype=float)
        p = p / max(float(p.sum()), EPS)
        status = local["reference_status"].astype(str).to_numpy()
        beneficial = status == "beneficial"
        harmful = status == "harmful"
        ambiguous = status == "ambiguous"
        clear = beneficial | harmful

        g0 = glob.iloc[0] if len(glob) else None
        clear_weight = float(np.sum(p * clear))
        ambiguous_weight = float(np.sum(p * ambiguous))
        testability = (
            "testable_on_clear_groups" if clear.any() else "pending_reference_precision"
        )

        rows.append(
            dict(
                campaign=campaign,
                direction=direction,
                n_local_groups=int(len(local)),
                n_clear_groups=int(clear.sum()),
                n_beneficial_groups=int(beneficial.sum()),
                n_harmful_groups=int(harmful.sum()),
                n_ambiguous_groups=int(ambiguous.sum()),
                clear_reference_weight=clear_weight,
                ambiguous_reference_weight=ambiguous_weight,
                mean_reference_ci_width=float(local["ref_ci_width"].mean()),
                median_reference_ci_width=float(local["ref_ci_width"].median()),
                weighted_mean_reference_ci_width=float(np.sum(p * local["ref_ci_width"].to_numpy(dtype=float))),
                total_random_n=int(g0["n_random"]) if g0 is not None else np.nan,
                total_bts_n=int(g0["n_bts"]) if g0 is not None else np.nan,
                total_random_clicks=int(g0["clicks_random"]) if g0 is not None else np.nan,
                total_bts_clicks=int(g0["clicks_bts"]) if g0 is not None else np.nan,
                global_delta_ref=float(g0["delta_ref"]) if g0 is not None else np.nan,
                global_ref_ci_lower=float(g0["ref_ci_lower"]) if g0 is not None else np.nan,
                global_ref_ci_upper=float(g0["ref_ci_upper"]) if g0 is not None else np.nan,
                global_reference_status=str(g0["reference_status"]) if g0 is not None else "unavailable",
                h8_reference_testability=testability,
                reference_interval_method=REFERENCE_INTERVAL_METHOD,
            )
        )
    return pd.DataFrame(rows)


def _evaluability_summary(diags: pd.DataFrame) -> pd.DataFrame:
    if diags.empty:
        return pd.DataFrame()

    local = diags[diags["group"] >= 0].copy()
    glob = diags[diags["group"] == -1].copy()
    keys = ["campaign", "direction", "n", "n_label"]

    rows = []
    for key_vals, d in local.groupby(keys, dropna=False):
        meta = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        gmatch = glob.copy()
        for k, v in meta.items():
            gmatch = gmatch[gmatch[k] == v]

        rows.append(
            {
                **meta,
                "n_local_group_records": int(len(d)),
                "mean_local_ess_fraction": float(d["ess_fraction"].mean()),
                "median_local_ess_fraction": float(d["ess_fraction"].median()),
                "mean_local_weighted_ess_fraction": float(d["weighted_ess_fraction"].mean()),
                "median_local_w_q99": float(d["w_q99"].median()),
                "max_local_w_max": float(d["w_max"].max()),
                "min_local_pscore": float(d["pscore_min"].min()),
                "global_ess_fraction": float(gmatch["ess_fraction"].mean()) if len(gmatch) else np.nan,
                "global_weighted_ess_fraction": float(gmatch["weighted_ess_fraction"].mean()) if len(gmatch) else np.nan,
                "global_w_q99": float(gmatch["w_q99"].mean()) if len(gmatch) else np.nan,
                "global_w_max": float(gmatch["w_max"].max()) if len(gmatch) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _h8_status_table(ref_precision: pd.DataFrame) -> pd.DataFrame:
    if ref_precision.empty:
        return pd.DataFrame()

    rows = []
    for _, r in ref_precision.iterrows():
        if int(r["n_clear_groups"]) == 0:
            state = "pending_reference_precision"
            interpretation = (
                "No clear local on-policy reference groups. Do not score H8, "
                "clear-harm exposure, or local sign agreement as evidence of safety."
            )
        else:
            state = "evaluable_on_clear_groups"
            interpretation = (
                "H8 may be scored only on clear beneficial/harmful reference groups; "
                "ambiguous groups remain excluded from sign-error claims."
            )
        rows.append(
            dict(
                campaign=r["campaign"],
                direction=r["direction"],
                h8_state=state,
                n_clear_groups=int(r["n_clear_groups"]),
                n_ambiguous_groups=int(r["n_ambiguous_groups"]),
                clear_reference_weight=float(r["clear_reference_weight"]),
                ambiguous_reference_weight=float(r["ambiguous_reference_weight"]),
                interpretation=interpretation,
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def _plot_campaign(campaign, ref_all, gate_all, point_all, decision_all):
    out = cfg.FIGURES_DIR / "obd"
    out.mkdir(parents=True, exist_ok=True)

    # Forest plot: LOCAL independent reference only. Global row is deliberately
    # excluded because it is a different estimand from the frozen local groups.
    ref = ref_all[
        (ref_all["campaign"] == campaign)
        & (ref_all["direction"] == "random_to_bts")
        & (ref_all["group"] >= 0)
    ].sort_values("delta_ref")
    if len(ref):
        fig, ax = plt.subplots(figsize=(7.5, max(5, 0.25 * len(ref))))
        y = np.arange(len(ref))
        x = ref["delta_ref"].to_numpy(dtype=float)
        xe = np.vstack(
            [
                x - ref["ref_ci_lower"].to_numpy(dtype=float),
                ref["ref_ci_upper"].to_numpy(dtype=float) - x,
            ]
        )
        ax.errorbar(x, y, xerr=xe, fmt="o")
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(ref["group"].astype(str))
        ax.set_xlabel("Independent on-policy BTS - Random CTR difference")
        ax.set_ylabel("Frozen context-position group")
        ax.set_title(f"OBD {campaign}: independent on-policy local reference")
        save_show(fig, out / f"fig_obd_{campaign}_reference_forest.png")

    # OPE vs reference scatter. Ambiguous points remain visible because this plot is
    # descriptive; H8 scoring still excludes ambiguous groups.
    p = point_all[
        (point_all["campaign"] == campaign)
        & (point_all["direction"] == "random_to_bts")
        & (point_all["estimator"] == "dr")
        & (point_all["group"] >= 0)
    ]
    if len(p) and len(ref):
        pp = (
            p.groupby("group", as_index=False)["estimate"]
            .mean()
            .merge(ref[["group", "delta_ref"]], on="group", how="inner")
        )
        if len(pp):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pp["delta_ref"], pp["estimate"])
            lo = min(pp["delta_ref"].min(), pp["estimate"].min())
            hi = max(pp["delta_ref"].max(), pp["estimate"].max())
            if np.isfinite(lo) and np.isfinite(hi):
                if math.isclose(lo, hi):
                    lo -= 1e-6
                    hi += 1e-6
                ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlabel("Independent on-policy reference")
            ax.set_ylabel("OPE DR estimate")
            ax.set_title(f"OBD {campaign}: OPE vs on-policy reference")
            save_show(fig, out / f"fig_obd_{campaign}_ope_vs_reference.png")

    # Evidence accumulation / direction comparison.
    d = decision_all[
        (decision_all["campaign"] == campaign)
        & (decision_all["method"] == "bb_dr")
    ].copy()
    if len(d):
        d["n_numeric"] = pd.to_numeric(d["n"], errors="coerce")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for direction, dd in d.groupby("direction"):
            tmp = (
                dd.groupby(["n_numeric", "n_label"], as_index=False)["deployment_coverage"]
                .mean()
                .sort_values("n_numeric")
            )
            ax.plot(
                tmp["n_label"].astype(str),
                tmp["deployment_coverage"],
                marker="o",
                label=direction,
            )
        ax.set_ylabel("Deployment coverage")
        ax.set_title(f"OBD {campaign}: evidence accumulation and negative control")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        save_show(fig, out / f"fig_obd_{campaign}_sample_size_gate.png")


# ============================================================================
# Internal self-test
# ============================================================================

def mock_obd_self_test():
    """No external OBP data required; catches fatal math/shape/reporting errors."""
    rng = np.random.default_rng(123)
    n = 800
    k = 8
    L = 3
    dx = 5
    dz = 4

    X = rng.normal(size=(n, dx))
    ac = rng.normal(size=(k, dz))
    pos = rng.integers(0, L, size=n)
    pi0 = np.full((L, k), 1 / k, dtype=float)
    raw = rng.gamma(2, 1, size=(L, k))
    pi1 = raw / raw.sum(axis=1, keepdims=True)
    a = np.array([rng.choice(k, p=pi0[p]) for p in pos])
    ps = np.full(n, 1 / k, dtype=float)
    logits = 0.2 * X[:, 0] + 0.3 * ac[a, 0] - 0.2 * pos
    r = rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float)

    comp = crossfit_obd_components(X, a, r, ps, pos, ac, pi0, pi1, 2, 123)
    if not np.isfinite(comp.dr_delta).all() or len(comp.dr_delta) != n:
        raise AssertionError("OBD mock DR failed.")

    comp_behavior_base = crossfit_obd_components(
        X, a, r, ps, pos, ac, pi0, pi1, 2, 124,
        baseline_is_behavior=True,
    )
    if not np.allclose(comp_behavior_base.w0, 1.0):
        raise AssertionError("OBD behavior-baseline w0 must equal one.")
    if not np.allclose(comp_behavior_base.dr0, r):
        raise AssertionError("OBD behavior-baseline DR0 must equal observed reward.")

    group = np.zeros(n, dtype=int)
    draws = dr_uncertainty_by_group(
        comp.dr_delta,
        group,
        50,
        124,
        bayesian=True,
    )
    if not np.isfinite(draws[0]).all():
        raise AssertionError("OBD mock uncertainty failed.")

    # Low-CTR Newcombe/Wilson reference test, including zero-click subgroups.
    rr = np.zeros(300, dtype=float)
    rb = np.zeros(320, dtype=float)
    rr[0] = 1.0
    rb[:2] = 1.0
    gr = np.repeat([0, 1], [150, 150])
    gb = np.repeat([0, 1], [160, 160])
    ref = _reference_table_from_rewards(rr, gr, rb, gb, "random_to_bts")
    if not np.isfinite(ref[["ref_ci_lower", "ref_ci_upper"]].to_numpy()).all():
        raise AssertionError("OBD low-CTR reference interval failed.")
    if -1 not in set(ref["group"]):
        raise AssertionError("OBD global reference row missing.")

    gate_df = pd.DataFrame({"group": [0, 1], "gate": [1, 0]})
    met = _reference_metrics(gate_df, ref)
    required_metric_keys = {
        "clear_harmful_exposure",
        "ambiguous_deployment_exposure",
        "reference_clear_policy_value",
        "reference_sign_selective_value",
    }
    if not required_metric_keys.issubset(met):
        raise AssertionError("OBD revised ambiguity-aware metrics missing.")
    if "reference_oracle_value" in met or "reference_oracle_regret" in met:
        raise AssertionError("OBD oracle terminology must not appear in revised metrics.")

    frontier = _reference_frontier(draws, ref)
    if frontier.empty or "ambiguous_deployment_exposure" not in frontier.columns:
        raise AssertionError("OBD revised frontier ambiguity accounting failed.")

    # Shared categorical basis test: each log alone would expose different levels.
    class _MiniDS:
        pass

    ds1 = _MiniDS()
    ds2 = _MiniDS()
    ds1.data = pd.DataFrame({"user_feature_0": ["A", "B", "A"]})
    ds2.data = pd.DataFrame({"user_feature_0": ["A", "C", "C"]})
    x1, x2 = _build_aligned_user_contexts(ds1, ds2)
    if x1.shape[1] != x2.shape[1]:
        raise AssertionError("OBD shared user-context basis self-test failed.")

    # Policy-specific anonymized hashes can differ while preserving the same
    # item-category partition.  The alignment routine must accept exactly that case.
    ds1.item_context = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "item_feature_0": [0.1, 0.2, 0.1, -0.3],
            "item_feature_1": ["rA", "rB", "rA", "rC"],
            "item_feature_2": ["rX", "rX", "rY", "rZ"],
            "item_feature_3": ["rM", "rN", "rM", "rN"],
        }
    )
    ds2.item_context = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "item_feature_0": [0.1, 0.2, 0.1, -0.3],
            "item_feature_1": ["b7", "b2", "b7", "b9"],
            "item_feature_2": ["q1", "q1", "q5", "q8"],
            "item_feature_3": ["z0", "z4", "z0", "z4"],
        }
    )
    ac_shared = _build_aligned_action_contexts(ds1, ds2, 4)
    if ac_shared.shape != (4, 4) or not np.isfinite(ac_shared).all():
        raise AssertionError("OBD shared action-context alignment self-test failed.")

    # A genuinely different partition must be rejected rather than silently aligned.
    ds_bad = _MiniDS()
    ds_bad.item_context = ds2.item_context.copy()
    ds_bad.item_context.loc[3, "item_feature_1"] = "b7"  # breaks the bijection
    rejected = False
    try:
        _build_aligned_action_contexts(ds1, ds_bad, 4)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError(
            "OBD action-context alignment failed to reject a genuine partition mismatch."
        )

    print(
        "[PASS] OBD internal mock self-test: action-context DR + BB uncertainty + "
        "shared user/action context bases + low-CTR reference + ambiguity-aware reporting"
    )


# ============================================================================
# Main runner
# ============================================================================

def run_obd():
    cfg.print_protocol()
    print("=" * 78)
    print(f"OBD RUNNER REVISION: {RUNNER_REVISION}")
    print(f"OBD METRICS VERSION : {OBD_METRICS_VERSION}")
    print(f"OBD TABLE OUTPUT DIR: {cfg.TABLES_DIR}")
    print("=" * 78)
    print("\n[RUN] Open Bandit Dataset real-log validation\n")

    # Clear only prior OBD outputs (never Synthetic/Semi outputs) so a failed run
    # cannot leave stale tables that look like fresh FINAL-R4 results.
    _clear_previous_obd_outputs()
    mock_obd_self_test()

    if os.environ.get("BCSPI_OBD_MOCK_ONLY", "0") == "1":
        print("BCSPI_OBD_MOCK_ONLY=1 -> external OBD loading intentionally skipped.")
        return None

    if cfg.RUN_MODE == "paper" and cfg.OBD_DATA_PATH is None:
        raise RuntimeError(
            "Paper mode requires FULL OBD. Set OBD_DATA_PATH in bcspi_config.py "
            "to the root containing random/all, bts/all, etc. The bundled 10k "
            "data are smoke-test only."
        )

    campaigns = cfg.OBD_CAMPAIGNS if cfg.RUN_MODE == "paper" else ["all"]

    ref_all = []
    point_all = []
    gate_all = []
    dec_all = []
    front_all = []
    diag_all = []
    temporal_all = []

    for ci, campaign in enumerate(campaigns):
        print(f"\n[OBD] Loading campaign={campaign} ...")
        dsr, fr, tr, dsb, fb, tb, random_pos, bts_pos = _load_pair(
            campaign,
            cfg.OBD_DATA_PATH,
        )
        print(
            f"  random n={fr['n_rounds']:,}, bts n={fb['n_rounds']:,}, "
            f"actions={fr['n_actions']}, positions={dsr.len_list}"
        )

        # Freeze one reward-blind group definition using Random contexts as the
        # design sample, then apply the same scaler/clusterer to the BTS reference.
        gr, gb, scaler, km, L = _fit_groups(
            np.asarray(fr["context"]),
            np.asarray(fb["context"]),
            np.asarray(fr["position"]),
            np.asarray(fb["position"]),
            cfg.RANDOM_SEED + ci,
        )

        for direction in ["random_to_bts", "bts_to_random"]:
            if direction == "random_to_bts":
                behavior, group, pi0, pi1 = fr, gr, random_pos, bts_pos
            else:
                behavior, group, pi0, pi1 = fb, gb, bts_pos, random_pos

            ref = _reference_table(fr, gr, fb, gb, direction)
            ref["campaign"] = campaign
            ref["direction"] = direction
            ref["metrics_version"] = OBD_METRICS_VERSION
            ref["runner_revision"] = RUNNER_REVISION
            ref_all.append(ref)

            n_total = int(behavior["n_rounds"])
            sizes = []
            for x in cfg.OBD_SAMPLE_SIZES:
                nn = n_total if x == "full" else min(int(x), n_total)
                if nn not in sizes:
                    sizes.append(nn)

            for nn in sizes:
                if nn == n_total:
                    reps = 1
                elif cfg.RUN_MODE == "paper":
                    reps = PAPER_REPS_PER_NONFULL_SIZE
                else:
                    reps = cfg.OBD_REPS_PER_SIZE
                for rep in range(reps):
                    seed = (
                        cfg.RANDOM_SEED
                        + ci * 1_000_000
                        + (0 if direction == "random_to_bts" else 500_000)
                        + nn
                        + rep
                    )
                    rng = np.random.default_rng(seed)
                    idx = (
                        np.arange(n_total)
                        if nn == n_total
                        else rng.choice(n_total, size=nn, replace=False)
                    )
                    print(
                        f"[OBD] {campaign} {direction} "
                        f"n={nn:,} rep={rep + 1}/{reps}"
                    )

                    comp, gsub, rsub, psub, point, diag, draws = _run_one_sample(
                        behavior,
                        group,
                        behavior["action_context"],
                        pi0,
                        pi1,
                        idx,
                        seed,
                        baseline_is_behavior=True,
                    )
                    n_label = "full" if nn == n_total else str(nn)
                    meta = dict(
                        layer="obd",
                        campaign=campaign,
                        direction=direction,
                        n=nn,
                        n_label=n_label,
                        rep=rep,
                        seed=seed,
                        metrics_version=OBD_METRICS_VERSION,
                        runner_revision=RUNNER_REVISION,
                    )

                    point = point.merge(
                        ref[["group", "delta_ref", "reference_status", "ref_ci_lower", "ref_ci_upper"]],
                        on="group",
                        how="left",
                    )
                    point["reference_error"] = point["estimate"] - point["delta_ref"]
                    for k, v in meta.items():
                        point[k] = v
                    point_all.append(point)

                    for k, v in meta.items():
                        diag[k] = v
                    diag_all.append(diag)

                    for method, dd in draws.items():
                        s = summaries_from_draw_dict(dd, cfg.ALPHA, cfg.DELTA)
                        s["method"] = method
                        s = s.merge(
                            ref[["group", "delta_ref", "reference_status", "ref_ci_lower", "ref_ci_upper"]],
                            on="group",
                            how="left",
                        )
                        s = s.merge(
                            diag[
                                [
                                    "group",
                                    "ess_pi1",
                                    "ess_fraction",
                                    "weighted_ess",
                                    "weighted_ess_fraction",
                                    "w_q99",
                                    "w_max",
                                    "pscore_min",
                                ]
                            ],
                            on="group",
                            how="left",
                        )
                        for k, v in meta.items():
                            s[k] = v
                        gate_all.append(s)

                        dec_all.append(
                            pd.DataFrame(
                                [
                                    {
                                        **meta,
                                        "method": method,
                                        **_reference_metrics(s[["group", "gate"]], ref),
                                    }
                                ]
                            )
                        )

                        ff = _reference_frontier(dd, ref)
                        if not ff.empty:
                            ff["method"] = method
                            for k, v in meta.items():
                                ff[k] = v
                            front_all.append(ff)

            # Optional temporal robustness: only after timestamp chronology is verified.
            if cfg.RUN_OBD_TEMPORAL_ROBUSTNESS:
                times = tr if direction == "random_to_bts" else tb
                if times is None:
                    warnings.warn(
                        f"{campaign}/{direction}: no timestamp available; "
                        "temporal robustness skipped"
                    )
                else:
                    ts = pd.to_datetime(times, errors="coerce")
                    if ts.isna().mean() > 0.01:
                        warnings.warn(
                            f"{campaign}/{direction}: timestamps not reliably parseable; "
                            "temporal robustness skipped"
                        )
                    else:
                        order = np.argsort(ts.to_numpy())
                        cut = int(0.70 * len(order))
                        early = order[:cut]

                        # Later independent reference uses later 30% of BOTH on-policy logs.
                        trdt = pd.to_datetime(tr, errors="coerce")
                        tbdt = pd.to_datetime(tb, errors="coerce")
                        if trdt.isna().any() or tbdt.isna().any():
                            warnings.warn(
                                f"{campaign}/{direction}: a reference timestamp could not "
                                "be parsed; temporal robustness skipped"
                            )
                        else:
                            ri = np.argsort(trdt.to_numpy())[int(0.70 * len(trdt)) :]
                            bi = np.argsort(tbdt.to_numpy())[int(0.70 * len(tbdt)) :]
                            ref_late = _reference_table_from_rewards(
                                np.asarray(fr["reward"])[ri],
                                gr[ri],
                                np.asarray(fb["reward"])[bi],
                                gb[bi],
                                direction,
                            )

                            normalized = np.linspace(0, 1, len(early))
                            schemes = [
                                x
                                for x in cfg.RECENCY_SCHEMES
                                if x["kind"] in {"full", "rolling", "exp"}
                            ]
                            for scheme in schemes:
                                rr = recency_weights(
                                    normalized,
                                    scheme,
                                    seed=cfg.RANDOM_SEED,
                                )
                                (
                                    comp,
                                    gsub,
                                    rsub,
                                    psub,
                                    point,
                                    diag,
                                    draws,
                                ) = _run_one_sample(
                                    behavior,
                                    group,
                                    behavior["action_context"],
                                    pi0,
                                    pi1,
                                    early,
                                    cfg.RANDOM_SEED + 77,
                                    base_weight=rr,
                                    baseline_is_behavior=True,
                                )

                                est_rows = []
                                for gg in sorted(np.unique(gsub)):
                                    mm = gsub == gg
                                    est_rows.append(
                                        {
                                            "group": int(gg),
                                            "estimate": float(
                                                np.sum(rr[mm] * comp.dr_delta[mm])
                                                / max(np.sum(rr[mm]), EPS)
                                            ),
                                        }
                                    )
                                estdf = pd.DataFrame(est_rows).merge(
                                    ref_late[
                                        ["group", "delta_ref", "reference_status"]
                                    ],
                                    on="group",
                                    how="inner",
                                )
                                clear = estdf["reference_status"] != "ambiguous"
                                mae = (
                                    float(
                                        np.mean(
                                            np.abs(
                                                estdf["estimate"] - estdf["delta_ref"]
                                            )
                                        )
                                    )
                                    if len(estdf)
                                    else np.nan
                                )
                                sign = (
                                    float(
                                        np.mean(
                                            (estdf.loc[clear, "estimate"] > 0)
                                            == (estdf.loc[clear, "delta_ref"] > 0)
                                        )
                                    )
                                    if clear.any()
                                    else np.nan
                                )
                                temporal_all.append(
                                    dict(
                                        campaign=campaign,
                                        direction=direction,
                                        relevance_scheme=scheme["name"],
                                        n_early=len(early),
                                        weighted_ess=float(
                                            diag.loc[
                                                diag["group"] == -1,
                                                "weighted_ess",
                                            ].iloc[0]
                                        ),
                                        dr_estimate=float(
                                            np.sum(rr * comp.dr_delta)
                                            / max(np.sum(rr), EPS)
                                        ),
                                        later_reference_mae=mae,
                                        later_clear_sign_agreement=sign,
                                        metrics_version=OBD_METRICS_VERSION,
                                        runner_revision=RUNNER_REVISION,
                                    )
                                )

        # Campaign plots after both directions are available.
        refdf = pd.concat(ref_all, ignore_index=True)
        ptdf = pd.concat(point_all, ignore_index=True)
        gtdf = pd.concat(gate_all, ignore_index=True)
        dcdf = pd.concat(dec_all, ignore_index=True)
        _plot_campaign(campaign, refdf, gtdf, ptdf, dcdf)

    # ------------------------------------------------------------------------
    # Consolidate and save long-form result tables
    # ------------------------------------------------------------------------
    refs = pd.concat(ref_all, ignore_index=True)
    points = pd.concat(point_all, ignore_index=True)
    gates = pd.concat(gate_all, ignore_index=True)
    decisions = pd.concat(dec_all, ignore_index=True)
    frontiers = (
        pd.concat(front_all, ignore_index=True) if front_all else pd.DataFrame()
    )
    diags = pd.concat(diag_all, ignore_index=True)

    save_csv(refs, cfg.TABLES_DIR / "obd_onpolicy_reference.csv")
    save_csv(points, cfg.TABLES_DIR / "obd_point_estimates_long.csv")
    save_csv(gates, cfg.TABLES_DIR / "obd_gate_results_long.csv")
    save_csv(decisions, cfg.TABLES_DIR / "obd_decision_results.csv")
    if not frontiers.empty:
        save_csv(frontiers, cfg.TABLES_DIR / "obd_safety_coverage_frontiers.csv")
    save_csv(diags, cfg.TABLES_DIR / "obd_evidence_diagnostics.csv")
    if temporal_all:
        save_csv(
            pd.DataFrame(temporal_all),
            cfg.TABLES_DIR / "obd_temporal_robustness.csv",
        )

    # ------------------------------------------------------------------------
    # Revised interpretation-safe summaries
    # ------------------------------------------------------------------------
    ref_summary = (
        refs[refs["group"] >= 0]
        .groupby(["campaign", "direction", "reference_status"], as_index=False)
        .size()
    )
    print_frame(
        "OBD independent on-policy LOCAL reference group counts",
        ref_summary,
        30,
    )

    ref_precision = _reference_precision_summary(refs)
    ref_precision["runner_revision"] = RUNNER_REVISION
    save_csv(
        ref_precision,
        cfg.TABLES_DIR / "obd_reference_precision_summary.csv",
    )
    print_frame(
        "OBD reference precision / H8 identifiability summary",
        ref_precision,
        30,
    )

    evaluability = _evaluability_summary(diags)
    evaluability["runner_revision"] = RUNNER_REVISION
    save_csv(
        evaluability,
        cfg.TABLES_DIR / "obd_evaluability_summary.csv",
    )
    print_frame(
        "OBD evaluability asymmetry summary",
        evaluability,
        50,
    )

    dec_summary = (
        decisions.groupby(
            ["campaign", "direction", "n_label", "method"],
            as_index=False,
        )
        .agg(
            deployment_coverage=("deployment_coverage", "mean"),
            clear_harmful_exposure=("clear_harmful_exposure", "mean"),
            ambiguous_deployment_exposure=("ambiguous_deployment_exposure", "mean"),
            clear_beneficial_deployment_exposure=("clear_beneficial_deployment_exposure", "mean"),
            sign_agreement=("clear_sign_agreement", "mean"),
            hybrid_value=("reference_hybrid_value", "mean"),
            reference_clear_policy_value=("reference_clear_policy_value", "mean"),
            hybrid_minus_reference_clear_policy=("hybrid_minus_reference_clear_policy", "mean"),
            reference_sign_selective_value=("reference_sign_selective_value", "mean"),
            hybrid_minus_reference_sign_selective=("hybrid_minus_reference_sign_selective", "mean"),
            clear_reference_weight=("clear_reference_weight", "mean"),
            ambiguous_reference_weight=("ambiguous_reference_weight", "mean"),
        )
    )
    dec_summary["runner_revision"] = RUNNER_REVISION
    save_csv(dec_summary, cfg.TABLES_DIR / "obd_decision_summary.csv")
    print_frame(
        "OBD deployment / independent-reference summary (ambiguity-aware)",
        dec_summary,
        50,
    )

    h8_status = _h8_status_table(ref_precision)
    h8_status["runner_revision"] = RUNNER_REVISION
    save_csv(h8_status, cfg.TABLES_DIR / "obd_h8_evidence_status.csv")
    print_frame("OBD H8 evidence status", h8_status, 30)

    save_json(
        {
            "runner_revision": RUNNER_REVISION,
            "run_mode": cfg.RUN_MODE,
            "campaigns": campaigns,
            "data_path": (
                str(cfg.OBD_DATA_PATH)
                if cfg.OBD_DATA_PATH
                else "OBP bundled small data"
            ),
            "bts_n_sim": cfg.OBD_BTS_N_SIM,
            "reps_per_nonfull_size": (
                PAPER_REPS_PER_NONFULL_SIZE
                if cfg.RUN_MODE == "paper"
                else cfg.OBD_REPS_PER_SIZE
            ),
            "compute_efficient_paper_protocol": bool(cfg.RUN_MODE == "paper"),
            "temporal_robustness": cfg.RUN_OBD_TEMPORAL_ROBUSTNESS,
            "metrics_version": OBD_METRICS_VERSION,
            "reference_interval_method": REFERENCE_INTERVAL_METHOD,
            "action_context_alignment": "item_id_plus_bijective_categorical_relabeling",
            "obd_reward_nuisance": "crossfit_logistic_sgd",
            "q_pi_computation": "explicit_per_action_probability_average",
            "behavior_baseline": "logged_on_policy_w0_equals_1",
            "bts_target_definition": "OBP_ZOZOTOWN_prior_static_evaluation_distribution",
            "oracle_metrics_present": False,
            "ambiguity_aware_reporting": True,
        },
        cfg.RESULTS_DIR / "obd_run_manifest.json",
    )

    _verify_required_outputs()

    print("\nOBD run completed without fatal validation errors.")
    print(f"Runner revision: {RUNNER_REVISION}")
    print(f"Tables : {cfg.TABLES_DIR}")
    print(f"Figures: {cfg.FIGURES_DIR / 'obd'}")
    print(
        "Interpretation rule: clear_harmful_exposure=0 is NOT a safety claim when "
        "ambiguous_reference_weight is large."
    )

    return refs, points, gates, decisions, frontiers, diags


if __name__ == "__main__":
    run_obd()
