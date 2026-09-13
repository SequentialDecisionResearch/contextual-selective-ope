r"""
Compute contrast-specific kappa diagnostics for the BC-SPI paper.

Put this file in:
C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code

Run:
    python compute_kappa_delta.py

or open it in Spyder and press F5.

What this script does
---------------------
1) DOES NOT rerun the expensive OPE / bootstrap experiment.
2) Reconstructs the already-frozen Synthetic S1 and Semi-Synthetic logging designs
   from their saved seeds and existing gate-result tables.
3) Computes, group by group,

       kappa_hat_{Delta,g}
       = mean_i [ ((pi1(A_i|X_i)-pi0(A_i|X_i))/mu(A_i|X_i))^2 ]

   and a Rao-Blackwellized / action-integrated version,

       kappa_RB_{Delta,g}
       = mean_i sum_a (pi1(a|X_i)-pi0(a|X_i))^2 / mu(a|X_i).

4) Merges kappa with the SAME BB-DR rows used for ESS/n and interval width.
5) Writes publication-ready summary CSVs under bcspi_results/tables/.

It does not modify or overwrite the original paper-result CSV files.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# USER / PROJECT SETTINGS
# -------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path(
    r"C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code"
)

# Optional override, useful only for testing on another machine:
# set BCSPI_PROJECT_DIR=C:\some\other\path
PROJECT_DIR = Path(os.environ.get("BCSPI_PROJECT_DIR", str(DEFAULT_PROJECT_DIR))).resolve()

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        "BC-SPI project directory was not found:\n"
        f"  {PROJECT_DIR}\n\n"
        "Put this .py file in the BC-SPI project directory or set BCSPI_PROJECT_DIR."
    )

os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import the ORIGINAL project code. Importing these modules does not run the
# full experiments because their run_* calls are protected by __main__.
import bcspi_config as cfg
import run_synthetic as syn
import run_semi_synthetic as semi

TABLES_DIR = Path(cfg.TABLES_DIR)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SYN_GATE_FILE = TABLES_DIR / "synthetic_gate_results_long.csv"
SEMI_GATE_FILE = TABLES_DIR / "semi_gate_results_long.csv"
SEMI_EXISTING_SUMMARY = TABLES_DIR / "semi_good_poor_overlap_summary.csv"

# Expert-requested comparison:
SYN_LEVELS = ("good", "severe")
SEMI_VARIANT = "local_stress"
SEMI_REGIMES = ("good_overlap", "poor_overlap")
PRIMARY_GROUP_SCHEME = "10_20_70"

# Prefer the manuscript's n=20,000 S1 result if it exists in the saved tables.
# Otherwise use the largest S1 n actually present.
PREFERRED_SYN_N = 20_000

# If True, reconstruct BOTH fixed_population and outer_robustness rows so that
# the semi summary aligns with the current manuscript's aggregation.
# This is still only deterministic diagnostic reconstruction: no nuisance model,
# no bootstrap, no OPE rerun.
INCLUDE_SEMI_OUTER_ROBUSTNESS = False


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n  {path}\n\n"
            "This script needs the frozen gate-result CSVs from the paper run."
        )


def _weighted_group_rows(
    pi0: np.ndarray,
    pi1: np.ndarray,
    mu: np.ndarray,
    groups: np.ndarray,
    action: np.ndarray,
) -> pd.DataFrame:
    """Compute realized and Rao-Blackwellized kappa by group plus overall."""
    pi0 = np.asarray(pi0, float)
    pi1 = np.asarray(pi1, float)
    mu = np.asarray(mu, float)
    groups = np.asarray(groups, int)
    action = np.asarray(action, int)

    if not (pi0.shape == pi1.shape == mu.shape):
        raise ValueError("pi0, pi1, and mu must have identical shape.")
    if len(groups) != len(pi0) or len(action) != len(pi0):
        raise ValueError("groups/action length does not match policy arrays.")
    if np.any(mu <= 0):
        raise ValueError("Behavior probabilities must be strictly positive.")

    n = len(groups)
    row = np.arange(n)
    delta = pi1 - pi0

    # Realized plug-in requested by the reviewer:
    signed_ratio = delta[row, action] / mu[row, action]
    kappa_obs_i = signed_ratio ** 2

    # Rao-Blackwellized / action-integrated counterpart:
    kappa_rb_i = np.sum((delta ** 2) / mu, axis=1)

    abs_ratio = np.abs(signed_ratio)

    out = []
    group_values = sorted(np.unique(groups).tolist()) + [-1]
    for g in group_values:
        mask = np.ones(n, dtype=bool) if g == -1 else (groups == g)
        if not np.any(mask):
            continue
        out.append(
            {
                "group": int(g),
                "n_group_reconstructed": int(mask.sum()),
                "kappa_hat": float(np.mean(kappa_obs_i[mask])),
                "kappa_rb": float(np.mean(kappa_rb_i[mask])),
                "delta_rho_abs_mean": float(np.mean(abs_ratio[mask])),
                "delta_rho_abs_q95": float(np.quantile(abs_ratio[mask], 0.95)),
                "delta_rho_abs_q99": float(np.quantile(abs_ratio[mask], 0.99)),
                "delta_rho_abs_max": float(np.max(abs_ratio[mask])),
            }
        )
    return pd.DataFrame(out)


def _safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if len(x) else np.nan


def _save(df: pd.DataFrame, name: str) -> Path:
    path = TABLES_DIR / name
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")
    return path


# -------------------------------------------------------------------------
# SYNTHETIC S1
# -------------------------------------------------------------------------
def compute_synthetic_s1():
    print("\n" + "=" * 80)
    print("1/2  Reconstructing Synthetic S1 contrast-complexity diagnostics")
    print("=" * 80)

    _require(SYN_GATE_FILE)
    gates = pd.read_csv(SYN_GATE_FILE)

    required_cols = {
        "scenario", "level", "rep", "seed", "n", "method", "group",
        "ess_fraction", "interval_width"
    }
    missing = required_cols - set(gates.columns)
    if missing:
        raise RuntimeError(f"Synthetic gate table is missing columns: {sorted(missing)}")

    g = gates[
        (gates["scenario"] == "S1_overlap")
        & (gates["method"] == "bb_dr")
        & (gates["group"] >= 0)
        & (gates["level"].isin(SYN_LEVELS))
    ].copy()

    if g.empty:
        raise RuntimeError("No S1_overlap / bb_dr / good|severe rows were found.")

    available_n = sorted(pd.to_numeric(g["n"], errors="coerce").dropna().astype(int).unique())
    if PREFERRED_SYN_N in available_n:
        target_n = PREFERRED_SYN_N
    else:
        target_n = max(available_n)
        print(
            f"[WARNING] n={PREFERRED_SYN_N:,} is not in the frozen table; "
            f"using largest available n={target_n:,}."
        )

    g = g[g["n"].astype(int) == target_n].copy()

    # Reconstruct each unique logging replication from the seed already stored
    # in the frozen gate table. This avoids relying on a hard-coded rep count.
    keys = (
        g[["level", "rep", "seed", "n"]]
        .drop_duplicates()
        .sort_values(["level", "rep"])
        .reset_index(drop=True)
    )

    print(f"S1 n used       : {target_n:,}")
    print(f"S1 replications : {len(keys)} total across {list(SYN_LEVELS)}")

    rows = []
    t0 = time.time()

    for j, rec in enumerate(keys.itertuples(index=False), start=1):
        level = str(rec.level)
        rep = int(rec.rep)
        seed = int(rec.seed)
        n = int(rec.n)

        if level not in syn.SCENARIOS["S1_overlap"]:
            raise KeyError(f"S1 level '{level}' is not present in run_synthetic.SCENARIOS.")

        c = syn.SCENARIOS["S1_overlap"][level]
        rng = np.random.default_rng(seed)

        # This reproduces the logging-data portion of generate_standard_condition
        # without generating truth populations, rewards, bootstraps, or nuisance fits.
        X, group = syn.sample_contexts(n, c["log_group_weights"], 0.0, rng)
        q = syn.q_current(X, group, c["nonlinear_strength"])
        pi0, pi1 = syn.target_policies(q, group, c["target_variant"])
        mu = syn.behavior_policy(q, c["tau"], c["mismatch"], c["exploration_floor"])
        action = syn.sample_categorical(mu, rng)

        kd = _weighted_group_rows(pi0, pi1, mu, group, action)
        kd = kd[kd["group"] >= 0].copy()
        kd.insert(0, "n", n)
        kd.insert(0, "seed", seed)
        kd.insert(0, "rep", rep)
        kd.insert(0, "level", level)
        kd.insert(0, "scenario", "S1_overlap")
        kd.insert(0, "layer", "synthetic")
        rows.append(kd)

        if j % 50 == 0 or j == len(keys):
            print(f"  reconstructed {j}/{len(keys)} S1 replications")

    kappa = pd.concat(rows, ignore_index=True)

    # Merge onto EXACT manuscript bb_dr rows. If there were duplicate nuisance
    # rows, this merge would preserve the manuscript weighting automatically.
    merge_keys = ["level", "rep", "seed", "n", "group"]
    gm = g.merge(kappa, on=merge_keys, how="left", validate="many_to_one")

    if gm["kappa_hat"].isna().any():
        bad = int(gm["kappa_hat"].isna().sum())
        raise RuntimeError(f"Synthetic merge failed: {bad} BB-DR rows did not receive kappa.")

    summary = (
        gm.groupby(["level", "n"], as_index=False)
        .agg(
            kappa_hat_mean=("kappa_hat", "mean"),
            kappa_hat_sd=("kappa_hat", "std"),
            kappa_rb_mean=("kappa_rb", "mean"),
            kappa_rb_sd=("kappa_rb", "std"),
            ess_over_n_mean=("ess_fraction", "mean"),
            bbdr_width_mean=("interval_width", "mean"),
            delta_rho_abs_q99_mean=("delta_rho_abs_q99", "mean"),
            delta_rho_abs_max_mean=("delta_rho_abs_max", "mean"),
            n_bbdr_rows=("group", "size"),
        )
        .sort_values("level")
    )

    # Add the severe/good ratios when both levels exist.
    lookup = summary.set_index("level")
    if all(x in lookup.index for x in SYN_LEVELS):
        good = lookup.loc["good"]
        severe = lookup.loc["severe"]
        ratios = {
            "kappa_hat_severe_over_good": float(severe.kappa_hat_mean / good.kappa_hat_mean),
            "kappa_rb_severe_over_good": float(severe.kappa_rb_mean / good.kappa_rb_mean),
            "width_severe_over_good": float(severe.bbdr_width_mean / good.bbdr_width_mean),
        }
    else:
        ratios = {}

    print("\nSynthetic S1 publication summary:")
    print(summary.to_string(index=False))
    if ratios:
        print("\nSynthetic S1 severe/good ratios:")
        for k, v in ratios.items():
            print(f"  {k}: {v:.6f}")

    print(f"Synthetic reconstruction time: {time.time()-t0:.1f} sec")
    return kappa, gm, summary, ratios


# -------------------------------------------------------------------------
# SEMI-SYNTHETIC
# -------------------------------------------------------------------------
def _semi_u(phase: str, outer_seed: int, logging_draw: int, n: int) -> np.ndarray:
    """Reproduce the exact uniform draw used by run_semi_synthetic.py."""
    if phase == "fixed_population":
        seed = int(outer_seed) + 1000 + int(logging_draw)
    elif phase == "outer_robustness":
        seed = int(outer_seed) + int(logging_draw) + 999
    else:
        raise ValueError(f"Unsupported Semi phase: {phase}")
    return np.random.default_rng(seed).random(n)


def compute_semi():
    print("\n" + "=" * 80)
    print("2/2  Reconstructing Semi-Synthetic contrast-complexity diagnostics")
    print("=" * 80)

    _require(SEMI_GATE_FILE)
    gates = pd.read_csv(SEMI_GATE_FILE)

    required_cols = {
        "phase", "dataset", "candidate_variant", "regime", "group_scheme",
        "outer_seed", "logging_draw", "method", "group",
        "ess_fraction", "interval_width"
    }
    missing = required_cols - set(gates.columns)
    if missing:
        raise RuntimeError(f"Semi gate table is missing columns: {sorted(missing)}")

    phases = ["fixed_population"]
    if INCLUDE_SEMI_OUTER_ROBUSTNESS:
        phases.append("outer_robustness")

    g = gates[
        (gates["method"] == "bb_dr")
        & (gates["group"] >= 0)
        & (gates["candidate_variant"] == SEMI_VARIANT)
        & (gates["group_scheme"] == PRIMARY_GROUP_SCHEME)
        & (gates["regime"].isin(SEMI_REGIMES))
        & (gates["phase"].isin(phases))
    ].copy()

    if g.empty:
        raise RuntimeError("No matching Semi-Synthetic BB-DR rows were found.")

    # Population/model reconstruction is done once per (phase,dataset,outer_seed).
    pop_keys = (
        g[["phase", "dataset", "outer_seed"]]
        .drop_duplicates()
        .sort_values(["phase", "dataset", "outer_seed"])
        .reset_index(drop=True)
    )

    print(f"Semi populations to reconstruct : {len(pop_keys)}")
    print(f"Semi phases included            : {phases}")
    print("No bootstrap / nuisance fitting / OPE is performed.")

    rows = []
    t0 = time.time()

    for pj, pop in enumerate(pop_keys.itertuples(index=False), start=1):
        phase = str(pop.phase)
        dataset = str(pop.dataset)
        outer_seed = int(pop.outer_seed)

        X, y, pi0, candidates, groups, cuts = semi.prepare_population(
            dataset, outer_seed, PRIMARY_GROUP_SCHEME
        )
        pi1 = candidates[SEMI_VARIANT]

        # Unique logging draws/regimes actually present in the frozen gate table.
        need = (
            g[
                (g["phase"] == phase)
                & (g["dataset"] == dataset)
                & (g["outer_seed"].astype(int) == outer_seed)
            ][["regime", "logging_draw"]]
            .drop_duplicates()
            .sort_values(["regime", "logging_draw"])
        )

        # mu and RB kappa do not depend on logging_draw.
        mu_cache = {}
        rb_cache = {}
        for regime in need["regime"].unique():
            mu = semi.behavior_policy_from_pi0(pi0, semi.LOGGING_REGIMES[str(regime)])
            mu_cache[str(regime)] = mu

            # Dummy action is used only to call the helper; kappa_rb itself is
            # action-integrated. We store the group-specific RB values here.
            dummy_action = np.argmax(mu, axis=1)
            tmp = _weighted_group_rows(pi0, pi1, mu, groups, dummy_action)
            rb_cache[str(regime)] = tmp.set_index("group")["kappa_rb"].to_dict()

        for rec in need.itertuples(index=False):
            regime = str(rec.regime)
            draw = int(rec.logging_draw)
            mu = mu_cache[regime]

            u = _semi_u(phase, outer_seed, draw, len(y))
            action = semi.sample_with_uniform(mu, u)

            kd = _weighted_group_rows(pi0, pi1, mu, groups, action)
            kd = kd[kd["group"] >= 0].copy()

            # Replace with the cached action-integrated value (same number,
            # included explicitly for clarity).
            kd["kappa_rb"] = kd["group"].map(rb_cache[regime]).astype(float)

            kd.insert(0, "n", len(y))
            kd.insert(0, "logging_draw", draw)
            kd.insert(0, "outer_seed", outer_seed)
            kd.insert(0, "group_scheme", PRIMARY_GROUP_SCHEME)
            kd.insert(0, "regime", regime)
            kd.insert(0, "candidate_variant", SEMI_VARIANT)
            kd.insert(0, "dataset", dataset)
            kd.insert(0, "phase", phase)
            kd.insert(0, "layer", "semi")
            rows.append(kd)

        if pj % 5 == 0 or pj == len(pop_keys):
            print(f"  reconstructed {pj}/{len(pop_keys)} Semi populations")

    kappa = pd.concat(rows, ignore_index=True)

    merge_keys = [
        "phase", "dataset", "candidate_variant", "regime", "group_scheme",
        "outer_seed", "logging_draw", "group"
    ]
    gm = g.merge(kappa, on=merge_keys, how="left", validate="many_to_one")

    if gm["kappa_hat"].isna().any():
        bad = int(gm["kappa_hat"].isna().sum())
        ex = gm[gm["kappa_hat"].isna()][merge_keys].head(10)
        raise RuntimeError(
            f"Semi merge failed: {bad} BB-DR rows did not receive kappa.\n"
            f"Examples:\n{ex.to_string(index=False)}"
        )

    # A) Clean fixed-population summary: best for theory interpretation.
    fixed = gm[gm["phase"] == "fixed_population"].copy()
    fixed_summary = (
        fixed.groupby(["dataset", "regime"], as_index=False)
        .agg(
            kappa_hat_mean=("kappa_hat", "mean"),
            kappa_hat_sd=("kappa_hat", "std"),
            kappa_rb_mean=("kappa_rb", "mean"),
            kappa_rb_sd=("kappa_rb", "std"),
            ess_over_n_mean=("ess_fraction", "mean"),
            bbdr_width_mean=("interval_width", "mean"),
            delta_rho_abs_q99_mean=("delta_rho_abs_q99", "mean"),
            delta_rho_abs_max_mean=("delta_rho_abs_max", "mean"),
            n_bbdr_rows=("group", "size"),
        )
        .sort_values(["dataset", "regime"])
    )

    # B) Paper-aligned summary: same aggregation population as the manuscript's
    # existing semi_good_poor_overlap_summary.csv when outer robustness is included.
    aligned_summary = (
        gm.groupby(["dataset", "candidate_variant", "regime"], as_index=False)
        .agg(
            kappa_hat_mean=("kappa_hat", "mean"),
            kappa_hat_sd=("kappa_hat", "std"),
            kappa_rb_mean=("kappa_rb", "mean"),
            kappa_rb_sd=("kappa_rb", "std"),
            ess_over_n_mean=("ess_fraction", "mean"),
            bbdr_width_mean=("interval_width", "mean"),
            delta_rho_abs_q99_mean=("delta_rho_abs_q99", "mean"),
            delta_rho_abs_max_mean=("delta_rho_abs_max", "mean"),
            n_bbdr_rows=("group", "size"),
        )
        .sort_values(["dataset", "regime"])
    )

    print("\nSemi-Synthetic FIXED-POPULATION summary:")
    print(fixed_summary.to_string(index=False))

    print("\nSemi-Synthetic PAPER-ALIGNED summary:")
    print(aligned_summary.to_string(index=False))

    print(f"Semi reconstruction time: {time.time()-t0:.1f} sec")
    return kappa, gm, fixed_summary, aligned_summary


# -------------------------------------------------------------------------
# VALIDATION + PUBLICATION TABLE
# -------------------------------------------------------------------------
def build_publication_table(syn_summary, semi_aligned):
    syn_pub = syn_summary.copy()
    syn_pub.insert(0, "setting", "Synthetic S1")
    syn_pub.insert(1, "dataset", "")
    syn_pub = syn_pub.rename(columns={"level": "logging_regime"})
    syn_pub = syn_pub[
        [
            "setting", "dataset", "logging_regime",
            "kappa_hat_mean", "kappa_rb_mean",
            "ess_over_n_mean", "bbdr_width_mean",
            "delta_rho_abs_q99_mean", "delta_rho_abs_max_mean",
        ]
    ]

    semi_pub = semi_aligned.copy()
    semi_pub.insert(0, "setting", "Semi-Synthetic fixed local-stress")
    semi_pub = semi_pub.rename(columns={"regime": "logging_regime"})
    semi_pub = semi_pub[
        [
            "setting", "dataset", "logging_regime",
            "kappa_hat_mean", "kappa_rb_mean",
            "ess_over_n_mean", "bbdr_width_mean",
            "delta_rho_abs_q99_mean", "delta_rho_abs_max_mean",
        ]
    ]

    return pd.concat([syn_pub, semi_pub], ignore_index=True)


def validation_report(
    syn_merged,
    syn_summary,
    semi_merged,
    semi_aligned,
    syn_ratios,
):
    lines = []
    lines.append("BC-SPI kappa_delta validation report")
    lines.append("=" * 72)
    lines.append(f"Project: {PROJECT_DIR}")
    lines.append("")
    lines.append("Definition:")
    lines.append("kappa_hat = mean [ ((pi1(A|X)-pi0(A|X))/mu(A|X))^2 ]")
    lines.append("kappa_rb  = mean_X sum_a (pi1(a|X)-pi0(a|X))^2 / mu(a|X)")
    lines.append("")
    lines.append(f"Synthetic merged BB-DR rows: {len(syn_merged)}")
    lines.append(f"Semi merged BB-DR rows     : {len(semi_merged)}")
    lines.append(f"Synthetic missing kappa    : {int(syn_merged.kappa_hat.isna().sum())}")
    lines.append(f"Semi missing kappa         : {int(semi_merged.kappa_hat.isna().sum())}")
    lines.append("")

    # Theory-direction checks.
    if {"good", "severe"}.issubset(set(syn_summary["level"])):
        z = syn_summary.set_index("level")
        lines.append(
            "Synthetic S1 direction check: "
            f"kappa_hat severe/good = {z.loc['severe','kappa_hat_mean']/z.loc['good','kappa_hat_mean']:.6f}"
        )
        lines.append(
            "Synthetic S1 RB direction check: "
            f"kappa_rb severe/good = {z.loc['severe','kappa_rb_mean']/z.loc['good','kappa_rb_mean']:.6f}"
        )

    for dataset, d in semi_aligned.groupby("dataset"):
        zz = d.set_index("regime")
        if {"good_overlap", "poor_overlap"}.issubset(zz.index):
            lines.append(
                f"Semi {dataset}: kappa_hat poor/good = "
                f"{zz.loc['poor_overlap','kappa_hat_mean']/zz.loc['good_overlap','kappa_hat_mean']:.6f}; "
                f"kappa_rb poor/good = "
                f"{zz.loc['poor_overlap','kappa_rb_mean']/zz.loc['good_overlap','kappa_rb_mean']:.6f}"
            )

    lines.append("")
    lines.append("Alignment check against existing semi_good_poor_overlap_summary.csv:")
    if SEMI_EXISTING_SUMMARY.exists():
        old = pd.read_csv(SEMI_EXISTING_SUMMARY)
        old = old[old["candidate_variant"] == SEMI_VARIANT].copy()
        chk = semi_aligned.merge(
            old,
            on=["dataset", "candidate_variant", "regime"],
            how="inner",
            suffixes=("_new", "_old"),
        )
        if len(chk):
            ess_diff = np.max(np.abs(chk["ess_over_n_mean"] - chk["ess_fraction"]))
            width_diff = np.max(np.abs(chk["bbdr_width_mean"] - chk["interval_width"]))
            lines.append(f"max |ESS/n new-old| = {ess_diff:.12g}")
            lines.append(f"max |width new-old| = {width_diff:.12g}")
            if ess_diff < 1e-10 and width_diff < 1e-10:
                lines.append("PASS: paper-aligned aggregation exactly reproduces existing ESS/width summary.")
            else:
                lines.append(
                    "NOTE: aggregation differs from the existing summary. "
                    "Check whether the old summary mixed phases or came from another runner revision."
                )
        else:
            lines.append("No matching rows available for alignment.")
    else:
        lines.append("Existing semi summary file not found; alignment check skipped.")

    lines.append("")
    lines.append("Interpretation rule:")
    lines.append(
        "kappa is a contrast-specific overlap/evaluability coordinate, not a sufficient "
        "statistic for interval width. Theorem 4.3 also depends on n_g, extreme contrast "
        "weights, and nuisance precision."
    )
    return "\n".join(lines)


def main():
    started = time.time()

    print("=" * 80)
    print("BC-SPI THEORY-ALIGNED KAPPA_DIAGNOSTIC")
    print("=" * 80)
    print(f"Project dir : {PROJECT_DIR}")
    print(f"Tables dir  : {TABLES_DIR}")
    print("This is a deterministic diagnostic re-analysis, NOT a full experiment rerun.")
    print("Original paper outputs will NOT be overwritten.")

    syn_kappa, syn_merged, syn_summary, syn_ratios = compute_synthetic_s1()
    semi_kappa, semi_merged, semi_fixed, semi_aligned = compute_semi()

    publication = build_publication_table(syn_summary, semi_fixed)

    # Save detailed outputs.
    _save(syn_kappa, "kappa_delta_synthetic_s1_group_long.csv")
    _save(syn_summary, "kappa_delta_synthetic_s1_summary.csv")

    _save(semi_kappa, "kappa_delta_semi_group_long.csv")
    _save(semi_fixed, "kappa_delta_semi_fixed_population_summary.csv")
    _save(semi_aligned, "kappa_delta_semi_paper_aligned_summary.csv")

    _save(publication, "kappa_delta_main_publication_table.csv")

    report = validation_report(
        syn_merged=syn_merged,
        syn_summary=syn_summary,
        semi_merged=semi_merged,
        semi_aligned=semi_aligned,
        syn_ratios=syn_ratios,
    )
    report_path = TABLES_DIR / "kappa_delta_validation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")

    manifest = {
        "project_dir": str(PROJECT_DIR),
        "synthetic_gate_file": str(SYN_GATE_FILE),
        "semi_gate_file": str(SEMI_GATE_FILE),
        "preferred_synthetic_n": PREFERRED_SYN_N,
        "actual_synthetic_n": sorted(syn_summary["n"].astype(int).unique().tolist()),
        "synthetic_levels": list(SYN_LEVELS),
        "semi_variant": SEMI_VARIANT,
        "semi_regimes": list(SEMI_REGIMES),
        "semi_group_scheme": PRIMARY_GROUP_SCHEME,
        "include_semi_outer_robustness": INCLUDE_SEMI_OUTER_ROBUSTNESS,
        "runtime_seconds": float(time.time() - started),
        "note": (
            "No OPE, nuisance fitting, bootstrap, gate re-selection, or target-policy "
            "retuning was performed. Kappa was reconstructed from frozen seeds/policies."
        ),
    }
    manifest_path = TABLES_DIR / "kappa_delta_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SAVED] {manifest_path}")

    print("\n" + "=" * 80)
    print("PUBLICATION TABLE")
    print("=" * 80)
    print(publication.to_string(index=False))

    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    print(report)

    print("\nDONE.")
    print(f"Total runtime: {time.time()-started:.1f} sec")
    print("\nPlease send me these two files after the run:")
    print("  1) bcspi_results\\tables\\kappa_delta_main_publication_table.csv")
    print("  2) bcspi_results\\tables\\kappa_delta_validation_report.txt")


if __name__ == "__main__":
    main()
