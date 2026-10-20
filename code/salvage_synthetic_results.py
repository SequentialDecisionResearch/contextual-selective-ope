"""One-time recovery of completed Synthetic S0--S5 Paper results from checkpoint shards.

IMPORTANT
---------
This is NOT the normal Synthetic runner.  It performs NO OPE fitting and NO Monte Carlo
simulation.  It only reads the already-completed ``std__*.pkl`` checkpoint shards left by
the interrupted long run, validates that S0--S5 are complete, and rebuilds the normal
Synthetic CSV/figure outputs.

It never deletes or changes the checkpoint cache.  It also never touches Semi-Synthetic
or OBD outputs.

Expected project root:
    C:\\study_notes\\traval_rec\\Contextual_bandit_bayesian_ope_safe_imp\\BCSPI_python_code
Expected old checkpoint cache:
    C:\\study_notes\\traval_rec\\Contextual_bandit_bayesian_ope_safe_imp\\BCSPI_python_code\\bcspi_results\\cache\\synthetic_paper_checkpoint
"""
from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bcspi_config as cfg
from bcspi_core import (
    save_csv,
    save_json,
    save_show,
    print_frame,
    plot_calibration,
    plot_safety_coverage,
    monte_carlo_summary,
)

PROJECT_DIR = Path(cfg.PROJECT_DIR)
OLD_CHECKPOINT_DIR = Path(cfg.CACHE_DIR) / "synthetic_paper_checkpoint"

EXPECTED_N = [2_000, 5_000, 10_000, 20_000]
KEY_CONDITIONS = {
    ("S0_clean", "base"),
    ("S1_overlap", "severe"),
    ("S2_rare_heterogeneous", "severe"),
    ("S5_combined", "severe"),
}
EXPECTED_CONDITIONS = [
    ("S0_clean", "base"),
    *[("S1_overlap", x) for x in ["good", "moderate", "poor", "severe"]],
    *[("S2_rare_heterogeneous", x) for x in ["balanced", "mild", "rare", "severe"]],
    *[("S2_rare_homogeneous", x) for x in ["balanced", "mild", "rare", "severe"]],
    *[("S3_shift", x) for x in [
        "mean_0", "mean_025", "mean_05", "mean_10",
        "composition_mild", "composition_medium", "composition_severe",
    ]],
    *[("S4_misspecified", x) for x in ["linear", "mild_nonlinear", "strong_nonlinear"]],
    *[("S5_combined", x) for x in ["moderate", "severe"]],
]

BUCKETS = ["point", "gates", "decisions", "calibration", "frontiers", "diagnostics", "ratios"]


def _load_standard_shards(root: Path):
    shards = sorted(root.glob("std__*.pkl"))
    if not shards:
        raise RuntimeError(
            f"No standard Synthetic checkpoint shards found in:\n{root}\n"
            "Do NOT run the normal Synthetic runner. First verify the old checkpoint folder."
        )

    frames = {k: [] for k in BUCKETS}
    for i, path in enumerate(shards, 1):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise RuntimeError(f"Checkpoint shard is not a dict: {path.name}")
        for k in BUCKETS:
            df = obj.get(k)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames[k].append(df)
        if i % 100 == 0 or i == len(shards):
            print(f"[SALVAGE] loaded {i}/{len(shards)} standard checkpoint shards")

    out = {
        k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
        for k, v in frames.items()
    }
    return out, shards


def _validate_complete_s0_s5(point: pd.DataFrame):
    required_cols = {"scenario", "level", "n", "rep"}
    missing = required_cols - set(point.columns)
    if missing:
        raise RuntimeError(f"Checkpoint point table is missing columns: {sorted(missing)}")

    # Ignore any accidental S6 rows. This recovery is intentionally only S0--S5.
    p = point[point["scenario"] != "S6_temporal"].copy()
    observed_n = sorted(pd.to_numeric(p["n"], errors="raise").astype(int).unique().tolist())
    missing_n = sorted(set(EXPECTED_N) - set(observed_n))
    if missing_n:
        raise RuntimeError(f"S0--S5 recovery is incomplete: missing n values {missing_n}; observed={observed_n}")

    problems = []
    rows = []
    for n in EXPECTED_N:
        for scenario, level in EXPECTED_CONDITIONS:
            q = p[(p["n"] == n) & (p["scenario"] == scenario) & (p["level"] == level)]
            reps = int(q["rep"].nunique()) if len(q) else 0
            expected = 500 if (scenario, level) in KEY_CONDITIONS else 200
            rows.append({"n": n, "scenario": scenario, "level": level, "reps_found": reps, "reps_expected": expected})
            if reps < expected:
                problems.append((n, scenario, level, reps, expected))

    audit = pd.DataFrame(rows)
    if problems:
        head = "\n".join(
            f"  n={n} {s}/{l}: found {got}, expected {exp}"
            for n, s, l, got, exp in problems[:20]
        )
        raise RuntimeError(
            "Checkpoint S0--S5 set is not complete. No final files were overwritten.\n" + head
        )

    print("[PASS] All 25 S0--S5 conditions are complete at n=2k,5k,10k,20k.")
    print("[PASS] Key conditions have >=500 reps; all other conditions have >=200 reps.")
    return audit


def _backup_existing_synthetic_outputs():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = cfg.RESULTS_DIR / f"synthetic_pre_salvage_backup_{stamp}"
    tables_backup = backup / "tables"
    figs_backup = backup / "figures_synthetic"
    tables_backup.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in cfg.TABLES_DIR.glob("synthetic_*"):
        if f.is_file():
            shutil.copy2(f, tables_backup / f.name)
            copied += 1
    manifest = cfg.RESULTS_DIR / "synthetic_run_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, backup / manifest.name)
        copied += 1
    syn_fig = cfg.FIGURES_DIR / "synthetic"
    if syn_fig.exists() and any(syn_fig.iterdir()):
        shutil.copytree(syn_fig, figs_backup, dirs_exist_ok=True)
        copied += 1

    print(f"[BACKUP] Existing Synthetic final artifacts copied to: {backup}")
    print(f"[BACKUP] items copied: {copied}")
    return backup


def _clear_only_final_synthetic_outputs():
    for f in cfg.TABLES_DIR.glob("synthetic_*"):
        if f.is_file():
            f.unlink()
    manifest = cfg.RESULTS_DIR / "synthetic_run_manifest.json"
    if manifest.exists():
        manifest.unlink()
    syn_fig = cfg.FIGURES_DIR / "synthetic"
    if syn_fig.exists():
        shutil.rmtree(syn_fig)
    syn_fig.mkdir(parents=True, exist_ok=True)


def _plot_s0_s5(point, gates, decisions, calibration, frontiers):
    out = cfg.FIGURES_DIR / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    d = gates[(gates.scenario == "S0_clean") & (gates.method == "bb_dr") & (gates.group >= 0)]
    if len(d):
        t = d.groupby("group", as_index=False)["delta_true"].mean()
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.bar(t.group.astype(str), t.delta_true)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Subgroup")
        ax.set_ylabel("True local improvement")
        ax.set_title("Synthetic S0: global-good can contain local harm")
        save_show(fig, out / "fig_syn_truth_structure.png")

    c = calibration[(calibration.scenario == "S0_clean") & (calibration.group >= 0)]
    if len(c):
        cc = c.groupby(["method", "nominal"], as_index=False).covered.mean().rename(columns={"covered": "empirical"})
        plot_calibration(cc, "Synthetic S0 interval calibration", out / "fig_syn_s0_calibration.png")

    s = gates[(gates.scenario == "S1_overlap") & (gates.method == "bb_dr") & (gates.group >= 0)]
    if len(s):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(s.ess_fraction, s.interval_width, alpha=.6)
        ax.set_xlabel("Realized ESS / n")
        ax.set_ylabel("95% interval width")
        ax.set_title("S1: overlap evidence vs uncertainty")
        save_show(fig, out / "fig_syn_s1_ess_vs_width.png")

    s2 = decisions[decisions.scenario.str.startswith("S2_", na=False)]
    if len(s2):
        ss = s2.groupby(["scenario", "level", "method"], as_index=False)[["harmful_exposure", "deployment_coverage"]].mean()
        save_csv(ss, cfg.TABLES_DIR / "synthetic_s2_pooling_summary.csv")

    s3 = decisions[decisions.scenario == "S3_shift"]
    if len(s3):
        fig, ax = plt.subplots(figsize=(8, 5))
        last_tmp = None
        for m, dd in s3.groupby("method"):
            tmp = dd.groupby("level", as_index=False).harmful_exposure.mean()
            last_tmp = tmp
            ax.plot(np.arange(len(tmp)), tmp.harmful_exposure, marker="o", label=m)
        if last_tmp is not None:
            ax.set_xticks(np.arange(len(last_tmp)))
            ax.set_xticklabels(last_tmp.level, rotation=45, ha="right")
        ax.set_ylabel("Harmful exposure")
        ax.set_title("S3: shift and deployment harm")
        ax.legend(fontsize=7)
        save_show(fig, out / "fig_syn_s3_shift_harm.png")

    s4 = point[(point.scenario == "S4_misspecified") & (point.group >= 0)]
    if len(s4):
        ss = s4.groupby(["level", "estimator"], as_index=False).sq_error.mean()
        ss["rmse"] = np.sqrt(ss.sq_error)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for m, dd in ss.groupby("estimator"):
            ax.plot(dd.level, dd.rmse, marker="o", label=m)
        ax.set_ylabel("RMSE")
        ax.set_title("S4: reward-model misspecification")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        save_show(fig, out / "fig_syn_s4_rmse.png")

    f = frontiers[frontiers.scenario == "S5_combined"]
    if len(f):
        ff = f.groupby(["method", "posterior_threshold"], as_index=False)[["deployment_coverage", "harmful_exposure"]].mean()
        plot_safety_coverage(ff, "S5 combined-stress Safety-Coverage Frontier", out / "fig_syn_s5_frontier.png")


def salvage():
    if cfg.PROJECT_DIR.resolve() != PROJECT_DIR:
        raise RuntimeError(
            f"bcspi_config.PROJECT_DIR is {cfg.PROJECT_DIR}, expected {PROJECT_DIR}.\n"
            "Install the clean bcspi_config.py first; do not change the old checkpoint folder."
        )
    cfg.ensure_directories()
    print("=" * 78)
    print("SYNTHETIC CHECKPOINT SALVAGE -- NO ESTIMATORS WILL BE RERUN")
    print(f"Project    : {PROJECT_DIR}")
    print(f"Checkpoint : {OLD_CHECKPOINT_DIR}")
    print("Recovery scope: completed S0--S5 only; all already-computed n values are retained.")
    print("=" * 78)

    if not OLD_CHECKPOINT_DIR.exists():
        raise RuntimeError(f"Old checkpoint directory does not exist: {OLD_CHECKPOINT_DIR}")

    data, shards = _load_standard_shards(OLD_CHECKPOINT_DIR)
    required = ["point", "gates", "decisions", "calibration", "frontiers", "diagnostics"]
    empty = [k for k in required if data[k].empty]
    if empty:
        raise RuntimeError(f"Checkpoint recovery tables are empty: {empty}")

    audit = _validate_complete_s0_s5(data["point"])

    # Only after completeness validation do we back up and overwrite old Synthetic final files.
    backup_dir = _backup_existing_synthetic_outputs()
    _clear_only_final_synthetic_outputs()

    # Deliberately retain all completed S0--S5 work, including n=20,000, because it is
    # already computed and statistically valid.  Future normal runs use the leaner protocol.
    point = data["point"][data["point"].scenario != "S6_temporal"].copy()
    gates = data["gates"][data["gates"].scenario != "S6_temporal"].copy()
    decisions = data["decisions"][data["decisions"].scenario != "S6_temporal"].copy()
    calibration = data["calibration"][data["calibration"].scenario != "S6_temporal"].copy()
    frontiers = data["frontiers"][data["frontiers"].scenario != "S6_temporal"].copy()
    diagnostics = data["diagnostics"][data["diagnostics"].scenario != "S6_temporal"].copy()
    ratios = data["ratios"].copy()
    if len(ratios) and "scenario" in ratios.columns:
        ratios = ratios[ratios.scenario != "S6_temporal"].copy()

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
        decisions,
        ["scenario", "level", "method"],
        ["deployment_coverage", "harmful_exposure", "selective_value", "oracle_regret"],
    )
    point_summary = monte_carlo_summary(
        point,
        ["scenario", "level", "estimator", "group"],
        ["error", "abs_error", "sq_error"],
    )
    save_csv(calib_summary, base / "synthetic_calibration_summary.csv")
    save_csv(dec_summary, base / "synthetic_decision_summary.csv")
    save_csv(point_summary, base / "synthetic_point_summary.csv")
    save_csv(audit, base / "synthetic_salvage_checkpoint_audit.csv")

    _plot_s0_s5(point, gates, decisions, calibration, frontiers)

    source_protocol = None
    protocol_path = OLD_CHECKPOINT_DIR / "protocol.json"
    if protocol_path.exists():
        try:
            source_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except Exception:
            source_protocol = {"warning": "protocol.json could not be parsed"}

    observed_n = sorted(pd.to_numeric(point["n"], errors="raise").astype(int).unique().tolist())
    manifest = {
        "run_mode": "paper",
        "recovered_from_checkpoint": True,
        "recovery_scope": "S0-S5 only",
        "s6_present": False,
        "source_checkpoint_dir": str(OLD_CHECKPOINT_DIR),
        "source_checkpoint_shards_used": len(shards),
        "source_protocol": source_protocol,
        "observed_n_values": observed_n,
        "final_point_rows": int(len(point)),
        "final_gate_rows": int(len(gates)),
        "backup_before_salvage": str(backup_dir),
        "note": (
            "S0-S5 were recovered without rerunning estimators. n=20000 is retained because "
            "it was already completed. The clean future run_synthetic.py uses a separate, leaner "
            "protocol and a different checkpoint directory."
        ),
    }
    save_json(manifest, cfg.RESULTS_DIR / "synthetic_run_manifest.json")

    print_frame("Recovered Synthetic S0 calibration summary", calib_summary[calib_summary.scenario == "S0_clean"], 30)
    print("\n[SALVAGE COMPLETE]")
    print(f"Observed n values : {observed_n}")
    print("S6 included       : NO")
    print(f"Tables            : {cfg.TABLES_DIR}")
    print(f"Figures           : {cfg.FIGURES_DIR / 'synthetic'}")
    print(f"Old checkpoint    : {OLD_CHECKPOINT_DIR}  (UNCHANGED)")
    print("No Synthetic estimator or Monte Carlo replication was rerun.")


if __name__ == "__main__":
    salvage()
