r"""
BC-SPI Final Article Analysis and OBD DR Audit
==============================================

Purpose
-------
This is a READ-ONLY final analysis layer for the completed BC-SPI experiments.

It uses only existing CSV/JSON results under:
    C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code\bcspi_results

It DOES NOT rerun:
    - run_synthetic.py
    - run_semi_synthetic.py
    - run_obdnew.py
    - any estimator fitting or Monte Carlo simulation

It produces:
    bcspi_results\final_analysis\
        final_article_evidence.csv
        final_article_evidence.md
        final_numeric_summary.csv
        obd_dr_empirical_audit.csv
        obd_dr_code_audit.csv
        obd_ess_safeguard_sensitivity.csv
        followup_decision.txt
        final_analysis_manifest.json
        figures\
            fig_final_01_global_vs_local_truth.png/.pdf
            fig_final_02_s0_calibration.png/.pdf
            fig_final_03_evaluability_ess_width.png/.pdf
            fig_final_04_s5_safety_coverage.png/.pdf
            fig_final_05_semi_transfer_harm.png/.pdf
            fig_final_06_obd_raw_vs_ess_safeguard.png/.pdf

The program also performs a STATIC source-code audit of:
    run_obdnew.py
    bcspi_core.py

The static audit does not modify those files.

Spyder
------
Place this file in:
    C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code

Then open it in Spyder and press F5.

Interpretation discipline
-------------------------
1. The 5% ESS screen is reported as an ablation/sensitivity, NOT silently promoted
   to a newly tuned primary method.
2. OBD independent on-policy logs are noisy references, not oracle truth.
3. If OBD DR materially disagrees with IPS/SNIPS and the independent reference,
   this script flags a targeted OBD-only diagnostic rerun before claiming real-log
   DR validation. It does not recommend rerunning Synthetic or Semi-Synthetic.
4. S6/H9 is marked "not tested in the recovered final Synthetic package" if the
   manifest says S6 is absent.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(r"C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code")
RESULTS_DIR = Path(r"C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code\bcspi_results")

if not PROJECT_DIR.exists():
    raise FileNotFoundError(f"BC-SPI project directory does not exist: {PROJECT_DIR}")
os.chdir(PROJECT_DIR)
TABLES_DIR = RESULTS_DIR / "tables"
FINAL_DIR = RESULTS_DIR / "final_analysis"
FIG_DIR = FINAL_DIR / "figures"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
ESS_ABLATION_GRID = [0.00, 0.01, 0.025, 0.05, 0.10]
PRIMARY_DISPLAY_ESS_ABLATION = 0.05

METHOD_LABELS = {
    "bb_dr": "BB-DR",
    "bootstrap_dr": "Bootstrap DR",
    "bootstrap_snips": "Bootstrap SNIPS",
    "bb_dm": "BB-DM",
    "dm": "DM",
    "ips": "IPS",
    "snips": "SNIPS",
    "dr": "DR",
    "switch_dr": "SWITCH-DR",
    "snips_asym_lcb": "SNIPS LCB",
    "bb_dr_independent": "Independent BB-DR",
    "hier_eb_dr": "Hierarchical EB-DR",
    "bb_dr_complete_pool": "Complete pooling",
    "bb_dr_shift_estimated": "Shift-aware DR (estimated)",
    "bb_dr_shift_oracle": "Shift-aware DR (oracle)",
}


# =============================================================================
# Generic helpers
# =============================================================================

def _need(name: str) -> Path:
    p = TABLES_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required result file is missing:\n  {p}\n"
            "This final analysis must be run after Synthetic, Semi-Synthetic, "
            "and OBD result tables have been assembled."
        )
    return p


def _read(name: str, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(_need(name), usecols=usecols, low_memory=False)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stream_filter(
    name: str,
    usecols: list[str],
    predicate: Callable[[pd.DataFrame], pd.Series],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for ch in pd.read_csv(
        _need(name),
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        mask = predicate(ch)
        if mask.any():
            out.append(ch.loc[mask].copy())
    if not out:
        return pd.DataFrame(columns=usecols)
    return pd.concat(out, ignore_index=True)


def _safe_mean(x) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def _safe_median(x) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{100.0 * float(x):.{digits}f}%"


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    p = FINAL_DIR / name
    df.to_csv(p, index=False)
    print(f"[saved] {p}")
    return p


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {png}")
    print(f"[figure] {pdf}")


def _markdown_table(df: pd.DataFrame) -> str:
    # Avoid requiring tabulate.
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            s = "" if pd.isna(v) else str(v)
            s = s.replace("|", "\\|").replace("\n", " ")
            vals.append(s)
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def _source_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


# =============================================================================
# Load manifests and compact result tables
# =============================================================================

def load_core_tables() -> dict[str, pd.DataFrame | dict]:
    print("\n[1/8] Loading compact final result tables...")

    manifests = {
        "synthetic": _read_json(RESULTS_DIR / "synthetic_run_manifest.json"),
        "semi": _read_json(RESULTS_DIR / "semi_run_manifest.json"),
        "obd": _read_json(RESULTS_DIR / "obd_run_manifest.json"),
    }

    tables = {
        "manifests": manifests,
        "syn_decision": _read(
            "synthetic_decision_results.csv",
            [
                "scenario", "level", "rep", "seed", "n", "nuisance_model", "method",
                "deployment_coverage", "harmful_exposure",
                "global_candidate_harmful_exposure",
                "oracle_deployment_coverage", "selective_value",
                "oracle_selective_value", "oracle_regret",
                "baseline_value", "candidate_value",
            ],
        ),
        "syn_decision_summary": _read("synthetic_decision_summary.csv"),
        "syn_cal_summary": _read("synthetic_calibration_summary.csv"),
        "syn_point_summary": _read("synthetic_point_summary.csv"),
        "semi_truth": _read("semi_evaluation_only_truth_summary.csv"),
        "semi_decision": _read(
            "semi_decision_results.csv",
            [
                "phase", "dataset", "candidate_variant", "regime", "group_scheme",
                "outer_seed", "logging_draw", "n", "nuisance_model", "method",
                "deployment_coverage", "harmful_exposure",
                "global_candidate_harmful_exposure",
                "selective_value", "oracle_selective_value", "oracle_regret",
                "baseline_value", "candidate_value",
            ],
        ),
        "semi_decision_summary": _read("semi_decision_summary.csv"),
        "semi_cal_summary": _read("semi_calibration_summary.csv"),
        "semi_overlap_summary": _read("semi_good_poor_overlap_summary.csv"),
        "obd_ref": _read("obd_onpolicy_reference.csv"),
        "obd_ref_precision": _read("obd_reference_precision_summary.csv"),
        "obd_points": _read("obd_point_estimates_long.csv"),
        "obd_gates": _read("obd_gate_results_long.csv"),
        "obd_decisions": _read("obd_decision_results.csv"),
        "obd_diag": _read("obd_evidence_diagnostics.csv"),
        "obd_eval": _read("obd_evaluability_summary.csv"),
    }
    return tables


# =============================================================================
# Synthetic long-table slices (streamed)
# =============================================================================

def load_synthetic_slices() -> dict[str, pd.DataFrame]:
    print("\n[2/8] Streaming only the Synthetic rows needed for article-level inference...")

    common = [
        "scenario", "level", "rep", "seed", "n", "nuisance_model",
        "method", "group", "delta_true", "estimate_draw_mean",
        "ess_fraction", "interval_width",
    ]

    h1 = _stream_filter(
        "synthetic_gate_results_long.csv",
        common,
        lambda d: d["method"].eq("bb_dr"),
    )

    s1 = h1[
        h1["scenario"].eq("S1_overlap")
        & (pd.to_numeric(h1["group"], errors="coerce") >= 0)
    ].copy()

    s0 = h1[h1["scenario"].eq("S0_clean")].copy()

    shift = _stream_filter(
        "synthetic_gate_results_long.csv",
        common,
        lambda d: d["scenario"].eq("S3_shift")
        & d["method"].isin(
            ["bb_dr", "bb_dr_shift_estimated", "bb_dr_shift_oracle"]
        )
        & (pd.to_numeric(d["group"], errors="coerce") >= 0),
    )

    # Local-group calibration only. The paper's Gate is local, so global rows should
    # not inflate the calibration assessment.
    s0_cal = _stream_filter(
        "synthetic_calibration_long.csv",
        ["scenario", "method", "group", "nominal", "covered"],
        lambda d: d["scenario"].eq("S0_clean")
        & (pd.to_numeric(d["group"], errors="coerce") >= 0),
        chunksize=300_000,
    )

    # Pre-specified misspecified nuisance: linear reward model, local groups only.
    s4_points = _stream_filter(
        "synthetic_point_estimates_long.csv",
        ["scenario", "level", "nuisance_model", "group", "estimator", "sq_error"],
        lambda d: d["scenario"].eq("S4_misspecified")
        & d["nuisance_model"].eq("linear")
        & (pd.to_numeric(d["group"], errors="coerce") >= 0),
        chunksize=250_000,
    )

    return {
        "h1": h1,
        "s0": s0,
        "s1": s1,
        "shift": shift,
        "s0_cal": s0_cal,
        "s4_points": s4_points,
    }


# =============================================================================
# Numeric evidence calculations
# =============================================================================

def synthetic_evidence(t: dict, sl: dict) -> dict:
    h1 = sl["h1"].copy()
    keys = ["scenario", "level", "rep", "seed", "n", "nuisance_model"]

    glob = (
        h1[h1["group"].eq(-1)]
        .groupby(keys, dropna=False)["delta_true"]
        .mean()
        .rename("global_delta")
    )
    loc = (
        h1[h1["group"].ge(0)]
        .groupby(keys, dropna=False)["delta_true"]
        .min()
        .rename("min_local_delta")
    )
    joined = pd.concat([glob, loc], axis=1).dropna()
    h1_rate = float(
        ((joined["global_delta"] > 0) & (joined["min_local_delta"] < 0)).mean()
    )

    s0_truth = (
        sl["s0"]
        .groupby("group", as_index=False)["delta_true"]
        .mean()
        .sort_values("group")
    )
    s0_global = float(
        s0_truth.loc[s0_truth["group"].eq(-1), "delta_true"].iloc[0]
    )
    s0_min_local = float(
        s0_truth.loc[s0_truth["group"].ge(0), "delta_true"].min()
    )

    cal = sl["s0_cal"].copy()
    cal95 = cal[np.isclose(pd.to_numeric(cal["nominal"], errors="coerce"), 0.95)]
    syn_cal95 = cal95.groupby("method")["covered"].mean().to_dict()

    s1 = sl["s1"].copy()
    s1["ess_fraction"] = pd.to_numeric(s1["ess_fraction"], errors="coerce")
    s1["interval_width"] = pd.to_numeric(s1["interval_width"], errors="coerce")
    s1 = s1.dropna(subset=["ess_fraction", "interval_width"])
    corr_by_n = (
        s1.groupby("n")
        .apply(
            lambda d: d[["ess_fraction", "interval_width"]].corr().iloc[0, 1],
            include_groups=False,
        )
        .to_dict()
    )

    s1_means = (
        s1[s1["level"].isin(["good", "severe"])]
        .groupby(["n", "level"])[["ess_fraction", "interval_width"]]
        .mean()
    )
    max_n = int(pd.to_numeric(s1["n"], errors="coerce").max())
    good = s1_means.loc[(max_n, "good")]
    severe = s1_means.loc[(max_n, "severe")]
    severe_good_width_ratio = float(
        severe["interval_width"] / max(good["interval_width"], EPS)
    )
    severe_good_ess_ratio = float(
        severe["ess_fraction"] / max(good["ess_fraction"], EPS)
    )

    dec = t["syn_decision"]
    bb = dec[dec["method"].eq("bb_dr")].copy()
    syn_global_harm = float(bb["global_candidate_harmful_exposure"].mean())
    syn_gate_harm = float(bb["harmful_exposure"].mean())
    syn_coverage = float(bb["deployment_coverage"].mean())
    syn_value_gain = float((bb["selective_value"] - bb["baseline_value"]).mean())

    s5 = bb[
        bb["scenario"].eq("S5_combined")
        & bb["level"].eq("severe")
    ]
    s5_global_harm = float(s5["global_candidate_harmful_exposure"].mean())
    s5_gate_harm = float(s5["harmful_exposure"].mean())
    s5_coverage = float(s5["deployment_coverage"].mean())
    s5_value_gain = float((s5["selective_value"] - s5["baseline_value"]).mean())

    # S4: pre-specified deliberately misspecified LINEAR nuisance, local groups only.
    s4 = sl["s4_points"]
    s4 = s4[s4["level"].eq("strong_nonlinear")]
    s4_rmse = {}
    for est in ["dm", "dr", "ips", "snips", "switch_dr"]:
        z = s4[s4["estimator"].eq(est)]
        if len(z):
            s4_rmse[est] = float(
                np.sqrt(max(float(pd.to_numeric(z["sq_error"], errors="coerce").mean()), 0.0))
            )

    # S2 pooling severe.
    p = dec[
        dec["scenario"].isin(["S2_rare_heterogeneous", "S2_rare_homogeneous"])
        & dec["level"].eq("severe")
        & dec["method"].isin(
            ["bb_dr_independent", "hier_eb_dr", "bb_dr_complete_pool"]
        )
    ]
    pool = (
        p.groupby(["scenario", "method"])[
            ["harmful_exposure", "deployment_coverage", "oracle_regret"]
        ]
        .mean()
    )

    def _pool_val(scenario: str, method: str, metric: str) -> float:
        try:
            return float(pool.loc[(scenario, method), metric])
        except Exception:
            return np.nan

    # S3 shift-aware versus standard BB-DR.
    sh = sl["shift"].copy()
    sh["abs_error"] = (
        pd.to_numeric(sh["estimate_draw_mean"], errors="coerce")
        - pd.to_numeric(sh["delta_true"], errors="coerce")
    ).abs()
    sh_mean = (
        sh.groupby(["level", "method"])[["abs_error", "interval_width"]]
        .mean()
    )
    shift_level = "mean_10" if ("mean_10", "bb_dr") in sh_mean.index else sh["level"].iloc[0]

    def _shift_val(method: str, metric: str) -> float:
        try:
            return float(sh_mean.loc[(shift_level, method), metric])
        except Exception:
            return np.nan

    # BB-DR versus ordinary bootstrap DR: quantify near-equivalence in decisions.
    syn_sum = t["syn_decision_summary"].copy()
    keys2 = ["scenario", "level"]
    a = syn_sum[syn_sum["method"].eq("bb_dr")][
        keys2 + ["harmful_exposure_mean", "deployment_coverage_mean", "oracle_regret_mean"]
    ]
    b = syn_sum[syn_sum["method"].eq("bootstrap_dr")][
        keys2 + ["harmful_exposure_mean", "deployment_coverage_mean", "oracle_regret_mean"]
    ]
    comp = a.merge(b, on=keys2, suffixes=("_bb", "_boot"))
    syn_bb_boot_harm_mad = float(
        np.mean(
            np.abs(
                comp["harmful_exposure_mean_bb"]
                - comp["harmful_exposure_mean_boot"]
            )
        )
    ) if len(comp) else np.nan
    syn_bb_boot_cov_mad = float(
        np.mean(
            np.abs(
                comp["deployment_coverage_mean_bb"]
                - comp["deployment_coverage_mean_boot"]
            )
        )
    ) if len(comp) else np.nan

    return {
        "h1_rate": h1_rate,
        "s0_truth": s0_truth,
        "s0_global": s0_global,
        "s0_min_local": s0_min_local,
        "syn_cal95": syn_cal95,
        "s1_corr_by_n": corr_by_n,
        "s1_means": s1_means,
        "s1_max_n": max_n,
        "s1_width_ratio": severe_good_width_ratio,
        "s1_ess_ratio": severe_good_ess_ratio,
        "syn_global_harm": syn_global_harm,
        "syn_gate_harm": syn_gate_harm,
        "syn_coverage": syn_coverage,
        "syn_value_gain": syn_value_gain,
        "s5_global_harm": s5_global_harm,
        "s5_gate_harm": s5_gate_harm,
        "s5_coverage": s5_coverage,
        "s5_value_gain": s5_value_gain,
        "s4_rmse": s4_rmse,
        "pool_hetero_ind_harm": _pool_val(
            "S2_rare_heterogeneous", "bb_dr_independent", "harmful_exposure"
        ),
        "pool_hetero_hier_harm": _pool_val(
            "S2_rare_heterogeneous", "hier_eb_dr", "harmful_exposure"
        ),
        "pool_hetero_complete_harm": _pool_val(
            "S2_rare_heterogeneous", "bb_dr_complete_pool", "harmful_exposure"
        ),
        "pool_homo_ind_regret": _pool_val(
            "S2_rare_homogeneous", "bb_dr_independent", "oracle_regret"
        ),
        "pool_homo_hier_regret": _pool_val(
            "S2_rare_homogeneous", "hier_eb_dr", "oracle_regret"
        ),
        "shift_level": shift_level,
        "shift_bb_error": _shift_val("bb_dr", "abs_error"),
        "shift_est_error": _shift_val("bb_dr_shift_estimated", "abs_error"),
        "shift_oracle_error": _shift_val("bb_dr_shift_oracle", "abs_error"),
        "shift_bb_width": _shift_val("bb_dr", "interval_width"),
        "shift_est_width": _shift_val("bb_dr_shift_estimated", "interval_width"),
        "syn_bb_boot_harm_mad": syn_bb_boot_harm_mad,
        "syn_bb_boot_cov_mad": syn_bb_boot_cov_mad,
    }


def semi_evidence(t: dict) -> dict:
    truth = t["semi_truth"].copy()
    primary = truth[truth["group_scheme"].eq("10_20_70")]

    truth_rows = []
    local_stress_success = 0
    natural_negative = []
    for (dataset, variant), d in primary.groupby(["dataset", "candidate_variant"]):
        global_rows = d[d["group"].eq(-1)]
        if len(global_rows):
            global_delta = float(global_rows["delta_true"].mean())
        else:
            loc = d[d["group"].ge(0)]
            p = loc["population_weight"].to_numpy(dtype=float)
            p = p / max(p.sum(), EPS)
            global_delta = float(np.sum(p * loc["delta_true"].to_numpy(dtype=float)))
        min_local = float(d[d["group"].ge(0)]["delta_true"].min())
        truth_rows.append(
            {
                "dataset": dataset,
                "candidate_variant": variant,
                "global_delta": global_delta,
                "min_local_delta": min_local,
            }
        )
        if variant == "local_stress" and global_delta > 0 and min_local < 0:
            local_stress_success += 1
        if variant == "natural" and global_delta > 0 and min_local < 0:
            natural_negative.append(dataset)

    truth_summary = pd.DataFrame(truth_rows)

    d = t["semi_decision"].copy()
    bb = d[
        d["phase"].eq("fixed_population")
        & d["group_scheme"].eq("10_20_70")
        & d["candidate_variant"].eq("local_stress")
        & d["method"].eq("bb_dr")
    ].copy()

    semi_global_harm = float(bb["global_candidate_harmful_exposure"].mean())
    semi_gate_harm = float(bb["harmful_exposure"].mean())
    semi_coverage = float(bb["deployment_coverage"].mean())
    semi_value_gain = float((bb["selective_value"] - bb["baseline_value"]).mean())

    by_dataset_regime = (
        bb.groupby(["dataset", "regime"])[
            [
                "global_candidate_harmful_exposure",
                "harmful_exposure",
                "deployment_coverage",
                "selective_value",
                "baseline_value",
            ]
        ]
        .mean()
        .reset_index()
    )

    # Local primary-scheme calibration only; do not let global rows inflate coverage.
    semi_cal_local = _stream_filter(
        "semi_calibration_long.csv",
        ["phase", "group_scheme", "method", "group", "nominal", "covered"],
        lambda x: x["phase"].eq("fixed_population")
        & x["group_scheme"].eq("10_20_70")
        & (pd.to_numeric(x["group"], errors="coerce") >= 0),
        chunksize=300_000,
    )
    c95 = semi_cal_local[
        np.isclose(pd.to_numeric(semi_cal_local["nominal"], errors="coerce"), 0.95)
    ]
    semi_cal95 = c95.groupby("method")["covered"].mean().to_dict()

    ov = t["semi_overlap_summary"].copy()
    # This compact file is already fixed-population BB-DR local summary.
    overlap_rows = []
    for (dataset, variant), g in ov.groupby(["dataset", "candidate_variant"]):
        if set(g["regime"]) >= {"good_overlap", "poor_overlap"}:
            good = g[g["regime"].eq("good_overlap")].iloc[0]
            poor = g[g["regime"].eq("poor_overlap")].iloc[0]
            overlap_rows.append(
                {
                    "dataset": dataset,
                    "candidate_variant": variant,
                    "good_ess_fraction": float(good["ess_fraction"]),
                    "poor_ess_fraction": float(poor["ess_fraction"]),
                    "good_interval_width": float(good["interval_width"]),
                    "poor_interval_width": float(poor["interval_width"]),
                    "width_ratio_poor_over_good": float(
                        poor["interval_width"] / max(good["interval_width"], EPS)
                    ),
                }
            )
    overlap = pd.DataFrame(overlap_rows)

    # BB-DR versus Bootstrap DR incremental decision difference.
    ss = t["semi_decision_summary"].copy()
    keys = ["phase", "dataset", "candidate_variant", "regime"]
    a = ss[ss["method"].eq("bb_dr")][
        keys + ["harmful_exposure_mean", "deployment_coverage_mean", "oracle_regret_mean"]
    ]
    b = ss[ss["method"].eq("bootstrap_dr")][
        keys + ["harmful_exposure_mean", "deployment_coverage_mean", "oracle_regret_mean"]
    ]
    comp = a.merge(b, on=keys, suffixes=("_bb", "_boot"))
    semi_bb_boot_harm_mad = float(
        np.mean(
            np.abs(
                comp["harmful_exposure_mean_bb"]
                - comp["harmful_exposure_mean_boot"]
            )
        )
    ) if len(comp) else np.nan
    semi_bb_boot_cov_mad = float(
        np.mean(
            np.abs(
                comp["deployment_coverage_mean_bb"]
                - comp["deployment_coverage_mean_boot"]
            )
        )
    ) if len(comp) else np.nan

    return {
        "truth_summary": truth_summary,
        "local_stress_success": local_stress_success,
        "natural_negative": natural_negative,
        "semi_global_harm": semi_global_harm,
        "semi_gate_harm": semi_gate_harm,
        "semi_coverage": semi_coverage,
        "semi_value_gain": semi_value_gain,
        "semi_by_dataset_regime": by_dataset_regime,
        "semi_cal95": semi_cal95,
        "semi_overlap": overlap,
        "semi_bb_boot_harm_mad": semi_bb_boot_harm_mad,
        "semi_bb_boot_cov_mad": semi_bb_boot_cov_mad,
    }


def _obd_full_local_points(t: dict) -> pd.DataFrame:
    p = t["obd_points"].copy()
    p = p[
        p["n_label"].astype(str).eq("full")
        & (pd.to_numeric(p["group"], errors="coerce") >= 0)
    ].copy()
    return p


def obd_empirical_audit(t: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    print("\n[3/8] Computing OBD reference agreement, ESS stress, and safeguard sensitivity...")

    p = _obd_full_local_points(t)
    rows = []
    for (campaign, direction, estimator), d in p.groupby(
        ["campaign", "direction", "estimator"], dropna=False
    ):
        clear = d["reference_status"].astype(str).ne("ambiguous")
        x = d[clear].copy()
        if x.empty:
            continue
        err = pd.to_numeric(x["reference_error"], errors="coerce")
        est = pd.to_numeric(x["estimate"], errors="coerce")
        ref = pd.to_numeric(x["delta_ref"], errors="coerce")
        rows.append(
            {
                "campaign": campaign,
                "direction": direction,
                "estimator": estimator,
                "n_clear_group_records": int(len(x)),
                "mae_vs_onpolicy_reference": float(err.abs().mean()),
                "signed_bias_vs_reference": float(err.mean()),
                "sign_agreement": float(((est > 0) == (ref > 0)).mean()),
                "mean_estimate": float(est.mean()),
                "mean_reference": float(ref.mean()),
            }
        )
    audit = pd.DataFrame(rows)
    _write_csv(audit, "obd_dr_empirical_audit.csv")

    # Full-data ESS.
    diag = t["obd_diag"].copy()
    diag_full = diag[
        diag["n_label"].astype(str).eq("full")
        & (pd.to_numeric(diag["group"], errors="coerce") >= 0)
    ]
    ess = (
        diag_full.groupby(["campaign", "direction"])
        .agg(
            mean_local_ess_fraction=("ess_fraction", "mean"),
            median_local_ess_fraction=("ess_fraction", "median"),
            median_w_q99=("w_q99", "median"),
            max_w_max=("w_max", "max"),
            min_pscore=("pscore_min", "min"),
        )
        .reset_index()
    )

    # Raw and ESS-screened full-data Gates.
    gates = t["obd_gates"].copy()
    gates = gates[
        gates["n_label"].astype(str).eq("full")
        & gates["method"].astype(str).eq("bb_dr")
        & (pd.to_numeric(gates["group"], errors="coerce") >= 0)
    ].copy()

    refs = t["obd_ref"].copy()
    refs = refs[pd.to_numeric(refs["group"], errors="coerce") >= 0].copy()

    sens_rows = []
    for (campaign, direction), g in gates.groupby(["campaign", "direction"]):
        r = refs[
            refs["campaign"].eq(campaign)
            & refs["direction"].eq(direction)
        ][["group", "population_weight", "reference_status"]].copy()
        z = g.merge(r, on="group", how="inner", suffixes=("", "_ref"))
        if z.empty:
            continue
        pop = pd.to_numeric(z["population_weight"], errors="coerce").to_numpy(float)
        pop = pop / max(float(np.nansum(pop)), EPS)
        raw_gate = pd.to_numeric(z["gate"], errors="coerce").fillna(0).to_numpy(int)
        essf = pd.to_numeric(z["ess_fraction"], errors="coerce").to_numpy(float)
        status = z["reference_status_ref"].astype(str).to_numpy()
        harmful = status == "harmful"
        beneficial = status == "beneficial"
        ambiguous = status == "ambiguous"

        for threshold in ESS_ABLATION_GRID:
            dgate = raw_gate * (np.nan_to_num(essf, nan=-np.inf) >= threshold)
            sens_rows.append(
                {
                    "campaign": campaign,
                    "direction": direction,
                    "ess_fraction_threshold": threshold,
                    "deployment_coverage": float(np.sum(pop * dgate)),
                    "clear_harmful_exposure": float(np.sum(pop * dgate * harmful)),
                    "clear_beneficial_exposure": float(np.sum(pop * dgate * beneficial)),
                    "ambiguous_deployment_exposure": float(np.sum(pop * dgate * ambiguous)),
                    "n_deployed_groups": int(dgate.sum()),
                    "clear_reference_weight": float(np.sum(pop * (harmful | beneficial))),
                    "ambiguous_reference_weight": float(np.sum(pop * ambiguous)),
                }
            )

    sensitivity = pd.DataFrame(sens_rows)
    _write_csv(sensitivity, "obd_ess_safeguard_sensitivity.csv")

    # Compact summary for final evidence.
    def _audit_value(campaign, direction, estimator, metric):
        z = audit[
            audit["campaign"].eq(campaign)
            & audit["direction"].eq(direction)
            & audit["estimator"].eq(estimator)
        ]
        return float(z[metric].iloc[0]) if len(z) else np.nan

    dr_maes = audit[audit["estimator"].eq("dr")]["mae_vs_onpolicy_reference"]
    ips_maes = audit[audit["estimator"].eq("ips")]["mae_vs_onpolicy_reference"]
    snips_maes = audit[audit["estimator"].eq("snips")]["mae_vs_onpolicy_reference"]

    mean_dr_mae = float(dr_maes.mean()) if len(dr_maes) else np.nan
    mean_ips_mae = float(ips_maes.mean()) if len(ips_maes) else np.nan
    mean_snips_mae = float(snips_maes.mean()) if len(snips_maes) else np.nan

    reverse_ess = ess[ess["direction"].eq("bts_to_random")]["mean_local_ess_fraction"]
    forward_ess = ess[ess["direction"].eq("random_to_bts")]["mean_local_ess_fraction"]

    raw = sensitivity[np.isclose(sensitivity["ess_fraction_threshold"], 0.0)]
    scr = sensitivity[
        np.isclose(
            sensitivity["ess_fraction_threshold"],
            PRIMARY_DISPLAY_ESS_ABLATION,
        )
    ]

    raw_reverse_harm = float(
        raw[raw["direction"].eq("bts_to_random")]["clear_harmful_exposure"].mean()
    )
    scr_reverse_harm = float(
        scr[scr["direction"].eq("bts_to_random")]["clear_harmful_exposure"].mean()
    )
    raw_forward_harm = float(
        raw[raw["direction"].eq("random_to_bts")]["clear_harmful_exposure"].mean()
    )
    scr_forward_coverage = float(
        scr[scr["direction"].eq("random_to_bts")]["deployment_coverage"].mean()
    )

    # Evidence accumulation: compare full versus the largest repeated non-full size
    # per campaign, not 10k vs 50k, because the early curve is not monotone.
    og = t["obd_gates"].copy()
    og = og[
        og["method"].eq("bb_dr")
        & og["direction"].eq("random_to_bts")
        & (pd.to_numeric(og["group"], errors="coerce") >= 0)
    ].copy()
    contraction_rows = []
    for campaign, d in og.groupby("campaign"):
        nvals = sorted(pd.to_numeric(d["n"], errors="coerce").dropna().unique())
        if len(nvals) < 2:
            continue
        full_n = max(nvals)
        previous_n = max([x for x in nvals if x < full_n])
        full_w = float(
            d[pd.to_numeric(d["n"], errors="coerce").eq(full_n)]["interval_width"].mean()
        )
        prev_w = float(
            d[pd.to_numeric(d["n"], errors="coerce").eq(previous_n)]["interval_width"].mean()
        )
        contraction_rows.append(
            {
                "campaign": campaign,
                "previous_n": int(previous_n),
                "full_n": int(full_n),
                "previous_mean_interval_width": prev_w,
                "full_mean_interval_width": full_w,
                "full_over_previous_width_ratio": full_w / max(prev_w, EPS),
            }
        )
    contraction = pd.DataFrame(contraction_rows)

    return (
        {
            "mean_dr_mae": mean_dr_mae,
            "mean_ips_mae": mean_ips_mae,
            "mean_snips_mae": mean_snips_mae,
            "dr_to_ips_mae_ratio": mean_dr_mae / max(mean_ips_mae, EPS),
            "mean_forward_ess": float(forward_ess.mean()) if len(forward_ess) else np.nan,
            "mean_reverse_ess": float(reverse_ess.mean()) if len(reverse_ess) else np.nan,
            "raw_reverse_harm": raw_reverse_harm,
            "screened_reverse_harm": scr_reverse_harm,
            "raw_forward_harm": raw_forward_harm,
            "screened_forward_coverage": scr_forward_coverage,
            "contraction": contraction,
        },
        audit,
        sensitivity,
    )


# =============================================================================
# Static OBD DR source audit
# =============================================================================

def audit_obd_source(obd_empirical: dict) -> pd.DataFrame:
    print("\n[4/8] Performing static OBD DR source-code audit...")

    run_path = PROJECT_DIR / "run_obdnew.py"
    core_path = PROJECT_DIR / "bcspi_core.py"

    rows: list[dict] = []

    def add(check_id, severity, status, finding, implication, action):
        rows.append(
            {
                "check_id": check_id,
                "severity": severity,
                "status": status,
                "finding": finding,
                "implication": implication,
                "recommended_action": action,
            }
        )

    if not run_path.exists():
        add(
            "SRC-00",
            "HIGH",
            "NOT_CHECKED",
            f"run_obdnew.py not found at {run_path}",
            "Static audit cannot verify the exact runner used.",
            "Place the final run_obdnew.py in the project root and rerun only this analysis.",
        )
        run_text = ""
    else:
        run_text = _source_text(run_path)
        try:
            ast.parse(run_text)
            add(
                "SRC-01",
                "INFO",
                "PASS",
                "run_obdnew.py parses successfully with Python AST.",
                "No syntax-level issue detected in the audited source.",
                "None.",
            )
        except SyntaxError as e:
            add(
                "SRC-01",
                "CRITICAL",
                "FAIL",
                f"run_obdnew.py syntax error: {e}",
                "The source on disk may not be the source that produced the result package.",
                "Resolve source/version mismatch before interpreting the audit.",
            )

    if not core_path.exists():
        add(
            "SRC-02",
            "HIGH",
            "NOT_CHECKED",
            f"bcspi_core.py not found at {core_path}",
            "The actual crossfit_obd_components implementation cannot be statically verified.",
            "Place the exact bcspi_core.py used for OBD in the project root.",
        )
        core_text = ""
    else:
        core_text = _source_text(core_path)
        try:
            ast.parse(core_text)
            add(
                "SRC-02",
                "INFO",
                "PASS",
                "bcspi_core.py parses successfully with Python AST.",
                "The DR core source is available for audit.",
                "None.",
            )
        except SyntaxError as e:
            add(
                "SRC-02",
                "CRITICAL",
                "FAIL",
                f"bcspi_core.py syntax error: {e}",
                "The source on disk cannot be safely audited.",
                "Resolve source/version mismatch.",
            )

    if core_text:
        canonical_dr = all(
            token in core_text
            for token in [
                "resid = reward - q_obs",
                "dr0 = q0 + w0 * resid",
                "dr1 = q1 + w1 * resid",
            ]
        )
        add(
            "DR-01",
            "INFO" if canonical_dr else "CRITICAL",
            "PASS" if canonical_dr else "FAIL",
            "Canonical DR residual formula found."
            if canonical_dr
            else "Canonical DR residual formula was not found verbatim.",
            "The algebraic DR correction itself is structurally correct."
            if canonical_dr
            else "A formula-level implementation error may exist.",
            "None." if canonical_dr else "Inspect crossfit_obd_components before any further OBD use.",
        )

        crossfit_ok = (
            "fold_id" in core_text
            and "fold_id[start:stop] != fold" in core_text
            and "fold_id[start:stop] == fold" in core_text
        )
        add(
            "DR-02",
            "INFO" if crossfit_ok else "HIGH",
            "PASS" if crossfit_ok else "WARN",
            "Out-of-fold training/prediction separation is present."
            if crossfit_ok
            else "Could not verify explicit held-out prediction logic.",
            "Cross-fitting protects against same-row reward-model reuse."
            if crossfit_ok
            else "Nuisance overfitting could contaminate DR.",
            "None." if crossfit_ok else "Verify cross-fitting manually.",
        )

        linear_binary = (
            "SGDRegressor" in core_text
            and 'loss="squared_error"' in core_text
            and "np.clip(model.predict" in core_text
        )
        add(
            "DR-03",
            "MEDIUM",
            "LIMITATION" if linear_binary else "NOT_DETECTED",
            "OBD reward nuisance uses squared-error SGD with linear context/item interactions "
            "and clips predictions to [0,1]."
            if linear_binary
            else "The expected squared-error SGD + clipping pattern was not found.",
            "This is a valid conditional-mean regression strategy, but the final OBD data show "
            "DM/DR much worse than IPS/SNIPS, so nuisance misspecification/optimization is a "
            "plausible finite-sample failure mode.",
            "For the targeted OBD diagnostic only, compare against at least one alternative "
            "cross-fitted nuisance model or logistic-loss model.",
        )

        expected_feature_shortcut = (
            "ez0 = pi0_by_pos @ action_context" in core_text
            and "ez1 = pi1_by_pos @ action_context" in core_text
            and "q0[te] = np.clip(model.predict(F0)" in core_text
            and "q1[te] = np.clip(model.predict(F1)" in core_text
        )
        add(
            "DR-04",
            "HIGH",
            "REQUIRES_CHECK" if expected_feature_shortcut else "NOT_DETECTED",
            "Policy value nuisance is computed by predicting at E[action_context] and then clipping."
            if expected_feature_shortcut
            else "Expected-action-context shortcut pattern was not detected.",
            "For a linear model before clipping, f(E[z]) = E[f(z)] is valid. After clipping, "
            "clip(E[f]) need not equal E[clip(f)]. If per-action raw predictions cross 0 or 1, "
            "q_pi can be biased even though the linear shortcut is otherwise algebraically valid.",
            "Targeted diagnostic: on a representative OBD sample, explicitly enumerate actions, "
            "clip each q(x,a), average under pi, and compare with the current expected-feature shortcut.",
        )

        fixed_pseudo_boot = (
            "def dr_uncertainty_by_group" in core_text
            and "_weighted_bootstrap_means(z[m]" in core_text
        )
        add(
            "DR-05",
            "MEDIUM",
            "LIMITATION" if fixed_pseudo_boot else "NOT_DETECTED",
            "Bootstrap/BB uncertainty resamples already-computed cross-fitted DR pseudo-outcomes; "
            "the nuisance model is not refit inside each bootstrap draw."
            if fixed_pseudo_boot
            else "Could not verify fixed-pseudo-outcome bootstrap.",
            "This is computationally efficient and calibrated well in clean Synthetic, but it may "
            "under-represent nuisance uncertainty in difficult real-log settings.",
            "Keep the method labeled Bayesian-bootstrap/generalized uncertainty; for OBD, use "
            "independent on-policy validation and evaluability safeguards rather than claiming an exact posterior.",
        )

        # Inspect only the OBD DR function, not generic helpers elsewhere in bcspi_core.py.
        mfun = re.search(
            r"def\s+crossfit_obd_components\s*\(.*?(?=\n# ={10,}|\Z)",
            core_text,
            flags=re.DOTALL,
        )
        obd_func_text = mfun.group(0) if mfun else core_text
        no_dr_clip = (
            "w1 = p1_obs / np.clip(pscore" in obd_func_text
            and "dr1 = q1 + w1 * resid" in obd_func_text
            and "np.minimum(w1" not in obd_func_text
            and "np.clip(w1" not in obd_func_text
        )
        add(
            "DR-06",
            "HIGH",
            "RISK_CONFIRMED" if no_dr_clip else "CHECK",
            "Core DR uses the raw importance ratio without clipping/stabilization."
            if no_dr_clip
            else "Could not determine whether DR importance ratios are stabilized.",
            "This is unbiased under ideal assumptions but can be extremely unstable when support is poor. "
            f"Final OBD reverse-direction mean local ESS is {_pct(obd_empirical['mean_reverse_ess'])}.",
            "Do not interpret raw reverse DR Gate as safe. A targeted OBD diagnostic should include "
            "a pre-specified stabilized/clipped-DR sensitivity or a robust alternative.",
        )

    if run_text:
        bts_reconstruction = (
            "BernoulliTS(" in run_text
            and "compute_batch_action_dist" in run_text
            and "bts_by_pos" in run_text
        )
        reverse_uses_bts_pi0 = (
            'direction == "random_to_bts"' in run_text
            and "behavior, group, pi0, pi1 = fb, gb, bts_pos, random_pos" in run_text
        )
        pscore_identity_check = bool(
            re.search(
                r"(pi0.*pscore|pscore.*pi0|w0.*(?:mean|max|min|quantile|diagnostic))",
                run_text,
                flags=re.IGNORECASE,
            )
        )

        add(
            "DR-07",
            "HIGH",
            "REQUIRES_CHECK"
            if (bts_reconstruction and reverse_uses_bts_pi0 and not pscore_identity_check)
            else "PASS_OR_NOT_APPLICABLE",
            "Reverse BTS->Random uses a reconstructed position-level BTS matrix as pi0, while "
            "the log supplies per-row behavior pscore; no explicit identity diagnostic between "
            "pi0(A|position) and logged pscore was found."
            if (bts_reconstruction and reverse_uses_bts_pi0 and not pscore_identity_check)
            else "No unverified reverse behavior-policy identity risk was detected by the static pattern check.",
            "If reconstructed BTS is not exactly the behavior policy that generated each logged row, "
            "the OPE estimand and the independent BTS on-policy reference are not perfectly aligned. "
            "This is especially important because the reverse negative control is where raw DR fails.",
            "Mandatory targeted diagnostic before claiming reverse DR validation: report the distribution of "
            "rho0 = pi0_reconstructed(A_i|position_i) / logged_pscore_i. It should be tightly concentrated near 1 "
            "if pi0 is intended to equal the behavior policy.",
        )

        action_alignment = (
            "_build_aligned_action_contexts" in run_text
            and "_validate_policy_matrix" in run_text
            and "position" in run_text
        )
        add(
            "SRC-03",
            "INFO",
            "PASS" if action_alignment else "WARN",
            "Action-context alignment, policy-matrix validation, and position-aware processing are present."
            if action_alignment
            else "Could not verify all alignment/position guards from source patterns.",
            "These guards reduce the chance that the OBD discrepancy is a simple action-index or position bug."
            if action_alignment
            else "Alignment remains a possible source of error.",
            "None." if action_alignment else "Audit item/action/position alignment manually.",
        )

        w0_saved = "ess_pi0" in run_text or "w0_q99" in run_text or "w0_max" in run_text
        add(
            "DR-08",
            "MEDIUM",
            "MISSING_DIAGNOSTIC" if not w0_saved else "PASS",
            "The OBD diagnostics save target/candidate w1 diagnostics but do not appear to save symmetric w0 diagnostics."
            if not w0_saved
            else "Symmetric baseline w0 diagnostics appear to be saved.",
            "Without w0 diagnostics, the current CSV package cannot post-hoc verify whether the baseline "
            "side of the DR contrast is stable or whether reconstructed behavior weights are near 1.",
            "Add w0/behavior-identity diagnostics in the targeted OBD audit rerun; no need to rerun Synthetic/Semi.",
        )

    # Empirical red flag: DR should not be silently trusted when it is orders of magnitude
    # worse than IPS/SNIPS against independent reference.
    ratio = obd_empirical.get("dr_to_ips_mae_ratio", np.nan)
    empirical_flag = np.isfinite(ratio) and ratio >= 5.0
    add(
        "EMP-01",
        "HIGH" if empirical_flag else "INFO",
        "RED_FLAG" if empirical_flag else "PASS",
        f"Across full-data OBD clear groups, mean DR MAE / IPS MAE = {_fmt(ratio, 1)}x.",
        "A large ratio is direct evidence that the current OBD DR nuisance/correction combination "
        "is not the strongest real-log estimator in this result package.",
        "Before a paper claims OBD DR validation, complete the targeted OBD-only diagnostics in DR-03/04/06/07/08.",
    )

    audit_df = pd.DataFrame(rows)
    _write_csv(audit_df, "obd_dr_code_audit.csv")
    return audit_df


# =============================================================================
# Final article evidence table
# =============================================================================

def build_final_evidence(
    t: dict,
    syn: dict,
    semi: dict,
    obd: dict,
    code_audit: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[5/8] Building final_article_evidence.csv aligned to the paper's research questions...")

    evidence: list[dict] = []

    def add(
        qid: str,
        claim: str,
        status: str,
        layer: str,
        primary_numeric_evidence: str,
        interpretation: str,
        paper_role: str,
        required_action: str,
        figure: str = "",
    ):
        evidence.append(
            {
                "question": qid,
                "article_claim": claim,
                "status": status,
                "evidence_layer": layer,
                "primary_numeric_evidence": primary_numeric_evidence,
                "interpretation": interpretation,
                "paper_role": paper_role,
                "required_action": required_action,
                "recommended_figure": figure,
            }
        )

    add(
        "Q1/H1",
        "Global policy improvement does not imply local deployment safety.",
        "STRONG_SUPPORT",
        "Synthetic + Semi-Synthetic",
        (
            f"Synthetic replications with global Delta>0 and at least one local Delta_g<0: "
            f"{_pct(syn['h1_rate'])}. S0: global={_fmt(syn['s0_global'],5)}, "
            f"min local={_fmt(syn['s0_min_local'],5)}. "
            f"Semi local-stress: {semi['local_stress_success']}/3 datasets reproduce the pattern; "
            f"natural negative subgroup datasets={','.join(semi['natural_negative']) or 'none'}."
        ),
        "This is the clearest empirical justification for context-local policy admission rather than global rollout.",
        "MAIN_CLAIM",
        "No new experiment required.",
        "fig_final_01_global_vs_local_truth",
    )

    bb95 = syn["syn_cal95"].get("bb_dr", np.nan)
    boot95 = syn["syn_cal95"].get("bootstrap_dr", np.nan)
    dm95 = syn["syn_cal95"].get("bb_dm", np.nan)
    add(
        "Q2",
        "In the clean controlled setting, the DR-based uncertainty scale is approximately calibrated.",
        "STRONG_SUPPORT_WITH_ENGINE_QUALIFICATION",
        "Synthetic S0",
        (
            f"95% empirical coverage: BB-DR={_pct(bb95)}, "
            f"Bootstrap-DR={_pct(boot95)}, BB-DM={_pct(dm95)}."
        ),
        "DR-based intervals track nominal coverage closely; Bayesian/DM uncertainty does not. "
        "The paper should claim calibrated DR-based uncertainty, not 'Bayesian is automatically calibrated.'",
        "MAIN_MECHANISM",
        "Keep frequentist calibration as a prerequisite for deployment.",
        "fig_final_02_s0_calibration",
    )

    corr_txt = ", ".join(
        f"n={int(k):,}: {_fmt(v,3)}"
        for k, v in sorted(syn["s1_corr_by_n"].items())
    )
    add(
        "Q3/H2",
        "Evidence quality and policy evaluability are distinct from policy performance.",
        "STRONG_SUPPORT",
        "Synthetic S1 + Semi-Synthetic overlap + OBD ESS",
        (
            f"Synthetic corr(ESS/n, interval width): {corr_txt}. "
            f"At n={syn['s1_max_n']:,}, severe/good ESS ratio={_fmt(syn['s1_ess_ratio'],3)}, "
            f"width ratio={_fmt(syn['s1_width_ratio'],3)}. "
            f"OBD mean local ESS: forward={_pct(obd['mean_forward_ess'])}, "
            f"reverse={_pct(obd['mean_reverse_ess'])}."
        ),
        "Low support produces wider/less trustworthy local evidence; the OBD reverse direction is an extreme real-log example.",
        "MAIN_CLAIM",
        "Retain ESS/weight tails as diagnostics; do not collapse them into a tuned scalar score.",
        "fig_final_03_evaluability_ess_width",
    )

    add(
        "Q4/H3",
        "Selective local gating reduces harmful exposure without collapsing coverage.",
        "STRONG_SUPPORT",
        "Synthetic + Semi-Synthetic",
        (
            f"Synthetic mean global-rollout harm={_pct(syn['syn_global_harm'])} -> "
            f"BB-DR Gate harm={_pct(syn['syn_gate_harm'],3)} with coverage={_pct(syn['syn_coverage'])}. "
            f"Semi local-stress global harm={_pct(semi['semi_global_harm'])} -> "
            f"Gate harm={_pct(semi['semi_gate_harm'],3)} with coverage={_pct(semi['semi_coverage'])}."
        ),
        "This is the principal decision-level success criterion, stronger than a small RMSE improvement.",
        "MAIN_CLAIM",
        "No new broad experiment required.",
        "fig_final_04_s5_safety_coverage",
    )

    add(
        "Q5",
        "Safety is achieved with non-trivial deployment coverage rather than by always abstaining.",
        "STRONG_SUPPORT",
        "Synthetic S5 + Semi-Synthetic",
        (
            f"S5 severe: global harm={_pct(syn['s5_global_harm'])}, "
            f"Gate harm={_pct(syn['s5_gate_harm'],3)}, coverage={_pct(syn['s5_coverage'])}, "
            f"selective-value gain vs baseline={_fmt(syn['s5_value_gain'],5)}. "
            f"Semi selective-value gain vs baseline={_fmt(semi['semi_value_gain'],5)}."
        ),
        "The Gate retains meaningful coverage and positive value rather than obtaining safety by C=0.",
        "MAIN_DECISION_RESULT",
        "Report coverage and harm together in every main table.",
        "fig_final_05_semi_transfer_harm",
    )

    add(
        "Q6/H5",
        "Hierarchical borrowing helps compatible rare groups but can overshrink heterogeneous harmful groups.",
        "CONDITIONAL_SUPPORT_NOT_PRIMARY",
        "Synthetic S2",
        (
            f"Heterogeneous severe harm: independent={_pct(syn['pool_hetero_ind_harm'],4)}, "
            f"hierarchical={_pct(syn['pool_hetero_hier_harm'],4)}, "
            f"complete pooling={_pct(syn['pool_hetero_complete_harm'],3)}. "
            f"Homogeneous severe oracle regret: independent={_fmt(syn['pool_homo_ind_regret'],6)}, "
            f"hierarchical={_fmt(syn['pool_homo_hier_regret'],6)}."
        ),
        "Pooling is a conditional extension, not a universally safer main algorithm component.",
        "APPENDIX_CONDITIONAL_FINDING",
        "Do not make hierarchical pooling a headline contribution.",
    )

    dm_rmse = syn["s4_rmse"].get("dm", np.nan)
    dr_rmse = syn["s4_rmse"].get("dr", np.nan)
    add(
        "Q7/H4/H6",
        "The core DR/local-Gate mechanism is robust to model stress, but shift-aware weighting is not a demonstrated improvement.",
        "MIXED_SUPPORT",
        "Synthetic S4/S5/S3",
        (
            f"S4 strong-nonlinear RMSE: DM={_fmt(dm_rmse,5)}, DR={_fmt(dr_rmse,5)}. "
            f"S5 severe Gate harm={_pct(syn['s5_gate_harm'],3)} at coverage={_pct(syn['s5_coverage'])}. "
            f"S3 {syn['shift_level']} mean abs error: standard={_fmt(syn['shift_bb_error'],5)}, "
            f"estimated shift-aware={_fmt(syn['shift_est_error'],5)}, "
            f"oracle shift-aware={_fmt(syn['shift_oracle_error'],5)}."
        ),
        "DR provides useful protection relative to DM, while the shift-aware extension does not earn a main-algorithm role.",
        "MAIN_CORE_PLUS_APPENDIX_NEGATIVE_RESULT",
        "Keep S3 shift-aware results as robustness/negative evidence; do not tune them to force a win.",
    )

    add(
        "Q8/H7",
        "The Synthetic mechanism transfers to real feature geometry.",
        "STRONG_SUPPORT",
        "Semi-Synthetic Digits/Wine/Breast Cancer",
        (
            f"All {semi['local_stress_success']}/3 local-stress datasets show global-positive/local-negative truth. "
            f"Natural candidate negative subgroup appears in {','.join(semi['natural_negative']) or 'none'}. "
            f"BB-DR 95% coverage across fixed-population Semi={_pct(semi['semi_cal95'].get('bb_dr', np.nan))}."
        ),
        "Real X/Y geometry does not destroy the central local-harm and selective-deployment phenomenon, "
        "although Semi calibration is less perfect than clean Synthetic.",
        "MAIN_EXTERNAL_TRANSFER",
        "Report the Semi undercoverage honestly as a limitation under degraded support.",
        "fig_final_05_semi_transfer_harm",
    )

    add(
        "Q9/H8",
        "Real-log OBD external validation is directionally useful, but raw BB-DR does not pass the reverse negative control.",
        "PARTIAL_SUPPORT_REQUIRES_TARGETED_OBD_AUDIT",
        "Full OBD + independent on-policy reference",
        (
            f"Mean full-data clear-group MAE: DR={_fmt(obd['mean_dr_mae'],5)}, "
            f"IPS={_fmt(obd['mean_ips_mae'],5)}, SNIPS={_fmt(obd['mean_snips_mae'],5)}; "
            f"DR/IPS MAE ratio={_fmt(obd['dr_to_ips_mae_ratio'],1)}x. "
            f"Raw forward clear harmful exposure={_pct(obd['raw_forward_harm'])}; "
            f"raw reverse={_pct(obd['raw_reverse_harm'])}."
        ),
        "Forward rollout is conservative, but the reverse BTS->Random Gate wrongly deploys a reference-harmful policy "
        "under very poor support. This prevents a clean claim that raw OBD BB-DR is already validated.",
        "CRITICAL_REAL_DATA_LIMITATION",
        "Complete the targeted OBD DR diagnostic/re-run described in obd_dr_code_audit.csv before claiming real-log DR validation.",
        "fig_final_06_obd_raw_vs_ess_safeguard",
    )

    contraction = obd["contraction"]
    if len(contraction):
        contraction_txt = "; ".join(
            f"{r.campaign}: width {r.previous_mean_interval_width:.4f}->{r.full_mean_interval_width:.4f}"
            for r in contraction.itertuples()
        )
        ratios = contraction["full_over_previous_width_ratio"].to_numpy(float)
        contracts = bool(np.all(ratios < 1.0))
    else:
        contraction_txt = "NA"
        contracts = False

    add(
        "Q10",
        "Evidence accumulation appears at large OBD sample sizes, but the entire finite-sample curve is not strictly monotone.",
        "PARTIAL_SUPPORT",
        "OBD repeated subsampling",
        contraction_txt,
        "Intervals contract from the largest repeated subsample to full data in each campaign, and forward deployment "
        "generally increases. Early sample-size points can be non-monotone, so avoid claiming a strict monotone law.",
        "MAIN_OR_APPENDIX_EVIDENCE_ACCUMULATION",
        "No broad rerun required; report the full sample-size curve rather than cherry-picking endpoints.",
    )

    add(
        "Q11",
        "Most decision improvement comes from local selective deployment/evaluability, not a unique BB-DR advantage over ordinary Bootstrap-DR.",
        "STRONG_EVIDENCE_FOR_REPOSITIONING",
        "Synthetic + Semi-Synthetic ablation",
        (
            f"Mean absolute BB-DR vs Bootstrap-DR decision difference: "
            f"Synthetic harm={_fmt(syn['syn_bb_boot_harm_mad'],6)}, "
            f"coverage={_fmt(syn['syn_bb_boot_cov_mad'],6)}; "
            f"Semi harm={_fmt(semi['semi_bb_boot_harm_mad'],6)}, "
            f"coverage={_fmt(semi['semi_bb_boot_cov_mad'],6)}."
        ),
        "The paper should emphasize contextual policy admission and calibrated local evidence, not claim that Bayesian bootstrap dominates bootstrap DR.",
        "POSITIONING_RESULT",
        "Reframe novelty accordingly; no new expensive experiment required.",
    )

    add(
        "Q12",
        "Selective policy value is preserved or improved while harmful exposure is controlled.",
        "STRONG_SUPPORT_SYNTHETIC_SEMI_PARTIAL_OBD",
        "Synthetic + Semi-Synthetic; OBD forward reference",
        (
            f"Mean BB-DR selective-value gain vs baseline: Synthetic={_fmt(syn['syn_value_gain'],5)}, "
            f"Semi local-stress={_fmt(semi['semi_value_gain'],5)}. "
            f"OBD 5% ESS-screened forward mean coverage={_pct(obd['screened_forward_coverage'])} "
            f"with clear harmful exposure remaining {_pct(obd['raw_forward_harm'])} in the raw forward Gate."
        ),
        "The value/safety trade-off is strong in layers with truth; real-log OBD should be described with independent-reference hybrid values and the DR caveat.",
        "MAIN_VALUE_RESULT",
        "Do not use the post-hoc 5% ESS screen as a secretly tuned primary rule; show it as sensitivity/ablation.",
    )

    s6_present = bool(t["manifests"]["synthetic"].get("s6_present", False))
    add(
        "H9/S6",
        "Historical relevance / temporal nonstationarity is a robustness extension, not a main contribution.",
        "NOT_TESTED_IN_FINAL_PACKAGE" if not s6_present else "AVAILABLE",
        "Synthetic manifest",
        f"s6_present={s6_present}",
        "The recovered final Synthetic package contains S0-S5 only. No temporal-relevance claim should be made from this final result package.",
        "OMIT_FROM_MAIN_TEXT",
        "Do not rerun S6 unless temporal relevance becomes essential to the paper.",
    )

    df = pd.DataFrame(evidence)
    _write_csv(df, "final_article_evidence.csv")

    md = (
        "# BC-SPI Final Article Evidence\n\n"
        + _markdown_table(df)
        + "\n\n"
        + "## Interpretation rule\n\n"
        + "Statuses are derived from the completed result package. "
          "A failure or limitation is retained as evidence about where the proposed "
          "method should be narrowed; results are not re-tuned to force all hypotheses to pass.\n"
    )
    p = FINAL_DIR / "final_article_evidence.md"
    p.write_text(md, encoding="utf-8")
    print(f"[saved] {p}")

    return df


# =============================================================================
# Publication-level figures
# =============================================================================

def figure_01_global_vs_local(syn: dict) -> None:
    d = syn["s0_truth"].copy()
    loc = d[d["group"].ge(0)].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar([str(int(x)) for x in loc["group"]], loc["delta_true"])
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(
        syn["s0_global"],
        linewidth=1.5,
        linestyle="--",
        label=f"Global Delta = {syn['s0_global']:.3f}",
    )
    ax.set_xlabel("Pre-defined local subgroup")
    ax.set_ylabel("True policy improvement")
    ax.set_title("Global improvement can conceal a harmful local subgroup")
    ax.legend(frameon=False)
    _save_figure(fig, "fig_final_01_global_vs_local_truth")


def figure_02_calibration(sl: dict) -> None:
    d = sl["s0_cal"].copy()
    d = (
        d.groupby(["method", "nominal"], as_index=False)["covered"]
        .mean()
        .rename(columns={"covered": "empirical"})
    )
    methods = ["bb_dr", "bootstrap_dr", "bootstrap_snips", "bb_dm"]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for method in methods:
        z = d[d["method"].eq(method)].sort_values("nominal")
        if z.empty:
            continue
        ax.plot(
            z["nominal"],
            z["empirical"],
            marker="o",
            linewidth=1.5,
            label=METHOD_LABELS.get(method, method),
        )
    x = np.linspace(0.50, 1.00, 100)
    ax.plot(x, x, linestyle="--", linewidth=1.0, label="Ideal calibration")
    ax.set_xlim(0.49, 1.005)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Synthetic S0: uncertainty calibration")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    _save_figure(fig, "fig_final_02_s0_calibration")


def figure_03_evaluability(sl: dict) -> None:
    d = sl["s1"].copy()
    d["ess_fraction"] = pd.to_numeric(d["ess_fraction"], errors="coerce")
    d["interval_width"] = pd.to_numeric(d["interval_width"], errors="coerce")
    d = d.dropna(subset=["ess_fraction", "interval_width"])

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for n, z in d.groupby("n"):
        z = z.sort_values("ess_fraction").copy()
        # Quantile bins produce a readable publication curve without plotting every MC point.
        try:
            z["qbin"] = pd.qcut(z["ess_fraction"], q=10, duplicates="drop")
            b = (
                z.groupby("qbin", observed=True)
                .agg(
                    ess_fraction=("ess_fraction", "median"),
                    interval_width=("interval_width", "median"),
                )
                .dropna()
                .sort_values("ess_fraction")
            )
        except Exception:
            b = z[["ess_fraction", "interval_width"]]
        ax.plot(
            b["ess_fraction"],
            b["interval_width"],
            marker="o",
            linewidth=1.5,
            label=f"n={int(n):,}",
        )
    ax.set_xlabel("Realized local ESS / n")
    ax.set_ylabel("Median 95% interval width")
    ax.set_title("Synthetic S1: lower evaluability produces wider local uncertainty")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    _save_figure(fig, "fig_final_03_evaluability_ess_width")


def figure_04_s5_frontier(t: dict) -> None:
    print("[figure preparation] Streaming S5 severe safety-coverage frontier...")
    usecols = [
        "scenario", "level", "n", "method", "posterior_threshold",
        "deployment_coverage", "harmful_exposure",
    ]
    f = _stream_filter(
        "synthetic_safety_coverage_frontiers.csv",
        usecols,
        lambda d: d["scenario"].eq("S5_combined")
        & d["level"].eq("severe")
        & d["method"].isin(["bb_dr", "bootstrap_dr", "bootstrap_snips", "snips_asym_lcb"]),
        chunksize=200_000,
    )
    if f.empty:
        return

    max_n = int(pd.to_numeric(f["n"], errors="coerce").max())
    f = f[pd.to_numeric(f["n"], errors="coerce").eq(max_n)]
    agg = (
        f.groupby(["method", "posterior_threshold"])[
            ["deployment_coverage", "harmful_exposure"]
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method, z in agg.groupby("method"):
        z = z.sort_values("deployment_coverage")
        ax.plot(
            z["deployment_coverage"],
            z["harmful_exposure"],
            marker="o",
            linewidth=1.5,
            label=METHOD_LABELS.get(method, method),
        )

    dec = t["syn_decision"]
    z = dec[
        dec["scenario"].eq("S5_combined")
        & dec["level"].eq("severe")
        & dec["method"].eq("bb_dr")
        & pd.to_numeric(dec["n"], errors="coerce").eq(max_n)
    ]
    if len(z):
        global_harm = float(z["global_candidate_harmful_exposure"].mean())
        oracle_cov = float(z["oracle_deployment_coverage"].mean())
        ax.scatter([1.0], [global_harm], marker="X", s=90, label="Global rollout")
        ax.scatter([0.0], [0.0], marker="s", s=70, label="Always baseline")
        ax.scatter([oracle_cov], [0.0], marker="*", s=120, label="Oracle selective")

    ax.set_xlabel("Deployment coverage")
    ax.set_ylabel("Harmful exposure")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=1.0, base=10)
    ax.set_title(f"Synthetic S5 severe: safety-coverage frontier (n={max_n:,})")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=9)
    _save_figure(fig, "fig_final_04_s5_safety_coverage")


def figure_05_semi_transfer(semi: dict) -> None:
    d = semi["semi_by_dataset_regime"].copy()
    if d.empty:
        return
    d["label"] = (
        d["dataset"].astype(str).str.replace("_", " ").str.title()
        + " - "
        + d["regime"].astype(str).str.replace("_overlap", "").str.replace("_", " ").str.title()
    )

    x = np.arange(len(d))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(
        x - width / 2,
        d["global_candidate_harmful_exposure"],
        width,
        label="Global rollout",
    )
    ax.bar(
        x + width / 2,
        d["harmful_exposure"],
        width,
        label="Selective BB-DR",
    )
    for i, row in d.reset_index(drop=True).iterrows():
        ax.text(
            i + width / 2,
            float(row["harmful_exposure"]) + 0.002,
            f"C={100*float(row['deployment_coverage']):.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=25, ha="right")
    ax.set_ylabel("Population harmful exposure")
    ax.set_title("Semi-Synthetic transfer: local Gate removes harm without zero coverage")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False)
    _save_figure(fig, "fig_final_05_semi_transfer_harm")


def figure_06_obd_safeguard(sensitivity: pd.DataFrame) -> None:
    raw = sensitivity[np.isclose(sensitivity["ess_fraction_threshold"], 0.0)].copy()
    safe = sensitivity[
        np.isclose(
            sensitivity["ess_fraction_threshold"],
            PRIMARY_DISPLAY_ESS_ABLATION,
        )
    ].copy()

    rows = []
    for campaign in ["all", "men", "women"]:
        for direction in ["random_to_bts", "bts_to_random"]:
            for label, src in [("Raw Gate", raw), ("ESS>=5% ablation", safe)]:
                z = src[
                    src["campaign"].eq(campaign)
                    & src["direction"].eq(direction)
                ]
                if z.empty:
                    continue
                r = z.iloc[0]
                rows.append(
                    {
                        "label": (
                            f"{campaign.title()} "
                            f"{'R->BTS' if direction == 'random_to_bts' else 'BTS->R'} "
                            f"{label}"
                        ),
                        "clear_beneficial": float(r["clear_beneficial_exposure"]),
                        "ambiguous": float(r["ambiguous_deployment_exposure"]),
                        "clear_harmful": float(r["clear_harmful_exposure"]),
                    }
                )
    d = pd.DataFrame(rows)
    if d.empty:
        return

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    left = np.zeros(len(d))
    for key, label in [
        ("clear_beneficial", "Clear beneficial deployment"),
        ("ambiguous", "Ambiguous deployment"),
        ("clear_harmful", "Clear harmful deployment"),
    ]:
        vals = d[key].to_numpy(float)
        ax.barh(y, vals, left=left, label=label)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Deployment coverage, decomposed by independent reference")
    ax.set_title("Full OBD: raw Gate vs 5% ESS safeguard ablation")
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False, loc="lower right")
    _save_figure(fig, "fig_final_06_obd_raw_vs_ess_safeguard")


# =============================================================================
# Numeric summary and follow-up decision
# =============================================================================

def write_numeric_summary(syn: dict, semi: dict, obd: dict) -> pd.DataFrame:
    rows = [
        ("Synthetic H1 global-positive/local-negative rate", syn["h1_rate"]),
        ("Synthetic S0 global Delta", syn["s0_global"]),
        ("Synthetic S0 minimum local Delta", syn["s0_min_local"]),
        ("Synthetic BB-DR mean harmful exposure", syn["syn_gate_harm"]),
        ("Synthetic BB-DR mean deployment coverage", syn["syn_coverage"]),
        ("Synthetic S5 severe harmful exposure", syn["s5_gate_harm"]),
        ("Synthetic S5 severe coverage", syn["s5_coverage"]),
        ("Semi local-stress global rollout harm", semi["semi_global_harm"]),
        ("Semi local-stress BB-DR harm", semi["semi_gate_harm"]),
        ("Semi local-stress BB-DR coverage", semi["semi_coverage"]),
        ("OBD full clear-group mean DR MAE", obd["mean_dr_mae"]),
        ("OBD full clear-group mean IPS MAE", obd["mean_ips_mae"]),
        ("OBD DR/IPS MAE ratio", obd["dr_to_ips_mae_ratio"]),
        ("OBD mean forward local ESS fraction", obd["mean_forward_ess"]),
        ("OBD mean reverse local ESS fraction", obd["mean_reverse_ess"]),
        ("OBD raw reverse clear harmful exposure", obd["raw_reverse_harm"]),
        ("OBD 5% ESS-screen reverse clear harmful exposure", obd["screened_reverse_harm"]),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value"])
    _write_csv(df, "final_numeric_summary.csv")
    return df


def write_followup_decision(
    evidence: pd.DataFrame,
    code_audit: pd.DataFrame,
    obd: dict,
) -> None:
    critical_obd = (
        np.isfinite(obd["dr_to_ips_mae_ratio"])
        and obd["dr_to_ips_mae_ratio"] >= 5.0
    )
    high_static = code_audit[
        code_audit["severity"].isin(["CRITICAL", "HIGH"])
        & code_audit["status"].isin(
            ["FAIL", "WARN", "REQUIRES_CHECK", "RISK_CONFIRMED", "RED_FLAG", "MISSING_DIAGNOSTIC"]
        )
    ]

    if critical_obd or len(high_static):
        verdict = "YES — ONE TARGETED OBD-ONLY DIAGNOSTIC/RERUN IS REQUIRED BEFORE CLAIMING REAL-LOG DR VALIDATION."
        details = """
Do NOT rerun Synthetic or Semi-Synthetic.

The required follow-up is narrow:
1. Behavior-policy identity audit:
   for each OBD direction, save rho0_i =
   pi0_reconstructed(A_i | position_i) / logged_pscore_i.
   In a direction where pi0 is intended to equal the behavior policy, rho0 should
   be tightly concentrated around 1.

2. q_pi calculation audit:
   on a representative held-out OBD sample, explicitly predict q(x,a) for every
   action, clip each per-action prediction if clipping is retained, then form
   sum_a pi(a|x) q(x,a). Compare this with the current expected-action-context
   shortcut. This directly tests the post-clipping non-commutativity risk.

3. Nuisance/stability diagnostic:
   compare the current squared-error SGD nuisance with at least one alternative
   cross-fitted probability/conditional-mean model, and include a pre-specified
   stabilized/clipped-DR sensitivity for the extreme reverse weights.

4. Save symmetric w0 and w1 diagnostics.

This is a targeted code/OBD validation run. It is NOT a request for another large
three-layer experiment suite.
"""
    else:
        verdict = "NO ADDITIONAL EXPENSIVE EXPERIMENT IS CURRENTLY MANDATORY."
        details = """
The existing result package is internally coherent enough for article drafting.
Any additional experiment should be justified by a specific reviewer-facing need.
"""

    text = (
        "BC-SPI FINAL FOLLOW-UP DECISION\n"
        "================================\n\n"
        + verdict
        + "\n"
        + textwrap.dedent(details).strip()
        + "\n\n"
        "Paper positioning now supported by the data:\n"
        "- Global improvement does not imply local safety.\n"
        "- Performance and evaluability are distinct.\n"
        "- Calibrated local uncertainty can drive selective deployment.\n"
        "- Real-log deployment needs an explicit evaluability check when support collapses.\n"
        "- BB-DR should not be claimed to uniformly dominate ordinary Bootstrap-DR.\n"
    )
    p = FINAL_DIR / "followup_decision.txt"
    p.write_text(text, encoding="utf-8")
    print(f"[saved] {p}")
    print("\n" + text)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("BC-SPI FINAL ARTICLE ANALYSIS — READ-ONLY POST-PROCESSING")
    print("=" * 88)
    print(f"PROJECT_DIR : {PROJECT_DIR}")
    print(f"TABLES_DIR  : {TABLES_DIR}")
    print(f"OUTPUT_DIR  : {FINAL_DIR}")

    if not TABLES_DIR.exists():
        raise FileNotFoundError(
            f"Result table folder does not exist: {TABLES_DIR}\n"
            "Expected project layout: <PROJECT_DIR>\\bcspi_results\\tables"
        )

    t = load_core_tables()
    sl = load_synthetic_slices()

    syn = synthetic_evidence(t, sl)
    semi = semi_evidence(t)
    obd, obd_empirical_df, sensitivity_df = obd_empirical_audit(t)
    code_audit = audit_obd_source(obd)

    evidence = build_final_evidence(t, syn, semi, obd, code_audit)
    write_numeric_summary(syn, semi, obd)

    print("\n[6/8] Generating publication-level figures...")
    figure_01_global_vs_local(syn)
    figure_02_calibration(sl)
    figure_03_evaluability(sl)
    figure_04_s5_frontier(t)
    figure_05_semi_transfer(semi)
    figure_06_obd_safeguard(sensitivity_df)

    print("\n[7/8] Writing follow-up decision...")
    write_followup_decision(evidence, code_audit, obd)

    print("\n[8/8] Writing analysis manifest...")
    manifest = {
        "analysis_revision": "BCSPI_FINAL_ARTICLE_ANALYSIS_R1",
        "project_dir": str(PROJECT_DIR),
        "results_dir": str(RESULTS_DIR),
        "read_only_inputs": True,
        "reran_estimators": False,
        "reran_monte_carlo": False,
        "synthetic_manifest": t["manifests"]["synthetic"],
        "semi_manifest": t["manifests"]["semi"],
        "obd_manifest": t["manifests"]["obd"],
        "ess_ablation_grid": ESS_ABLATION_GRID,
        "primary_display_ess_ablation": PRIMARY_DISPLAY_ESS_ABLATION,
        "important_note": (
            "ESS=5% is displayed as an ablation/sensitivity, not promoted post-hoc "
            "to a tuned primary algorithm."
        ),
        "outputs": {
            "evidence": "final_article_evidence.csv",
            "numeric_summary": "final_numeric_summary.csv",
            "obd_empirical_audit": "obd_dr_empirical_audit.csv",
            "obd_code_audit": "obd_dr_code_audit.csv",
            "obd_ess_sensitivity": "obd_ess_safeguard_sensitivity.csv",
            "followup": "followup_decision.txt",
            "figure_dir": "figures",
        },
    }
    p = FINAL_DIR / "final_analysis_manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {p}")

    # Final sanity guards.
    expected = [
        FINAL_DIR / "final_article_evidence.csv",
        FINAL_DIR / "obd_dr_empirical_audit.csv",
        FINAL_DIR / "obd_dr_code_audit.csv",
        FINAL_DIR / "obd_ess_safeguard_sensitivity.csv",
        FINAL_DIR / "followup_decision.txt",
    ]
    expected += [
        FIG_DIR / f"fig_final_0{i}_{stem}.png"
        for i, stem in [
            (1, "global_vs_local_truth"),
            (2, "s0_calibration"),
            (3, "evaluability_ess_width"),
            (4, "s5_safety_coverage"),
            (5, "semi_transfer_harm"),
            (6, "obd_raw_vs_ess_safeguard"),
        ]
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            "Final analysis completed calculations but expected outputs are missing:\n"
            + "\n".join(missing)
        )

    print("\n" + "=" * 88)
    print("FINAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 88)
    print(f"Evidence table : {FINAL_DIR / 'final_article_evidence.csv'}")
    print(f"OBD code audit : {FINAL_DIR / 'obd_dr_code_audit.csv'}")
    print(f"Follow-up      : {FINAL_DIR / 'followup_decision.txt'}")
    print(f"Figures        : {FIG_DIR}")
    print("No estimator, Monte Carlo experiment, Synthetic run, Semi run, or OBD run was executed.")


if __name__ == "__main__":
    main()
