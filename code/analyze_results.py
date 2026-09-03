"""Turn BC-SPI experiment outputs into article-level evidence diagnostics.

R6-compatible analysis layer for the PRE-PAPER-FINAL-R6-DRFIX OBD runner.

This script never reruns estimators. It reads saved long-form result records,
prints hypothesis-level summaries to the Spyder console, and writes compact CSV
/ TXT evidence reports under bcspi_results/tables.

Important R6 OBD semantics
--------------------------
* OBD has no counterfactual oracle truth.
* Independent Random/BTS on-policy logs are noisy references, not an oracle.
* Local reference groups are scored only when their robust two-sample reference
  interval classifies them as clearly beneficial or clearly harmful.
* Ambiguous reference groups are excluded from sign-error / safety claims.
* If no clear local reference group exists, H8 is
  ``pending_reference_precision`` and MUST NOT be scored as evidence of safety.
* Obsolete OBD oracle columns (reference_oracle_value, reference_oracle_regret,
  oracle_regret) are forbidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bcspi_config as cfg
from bcspi_core import save_csv, print_frame


EXPECTED_OBD_RUNNER_REVISION = "PRE-PAPER-FINAL-R6-DRFIX"
EXPECTED_OBD_METRICS_VERSION = "obd_eval_v6_drfixed"
FORBIDDEN_OBD_ORACLE_COLUMNS = {
    "reference_oracle_value",
    "reference_oracle_regret",
    "oracle_regret",
}


def _read(name: str):
    p = cfg.TABLES_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add(rows, hypothesis, status, metric, value, interpretation):
    rows.append(
        dict(
            hypothesis=hypothesis,
            status=status,
            metric=metric,
            value=value,
            interpretation=interpretation,
        )
    )


def _require_columns(df: pd.DataFrame, name: str, required) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"R6 analysis cannot use {name}: missing required columns {missing}."
        )


def _validate_r6_obd_outputs(
    obdd,
    obd_h8,
    obd_ref_precision,
    obd_evaluability,
    manifest,
) -> None:
    """Fail loudly if OBD outputs are stale or not PRE-PAPER-FINAL-R6-DRFIX."""
    any_obd = any(x is not None for x in [obdd, obd_h8, obd_ref_precision, obd_evaluability])
    if not any_obd:
        return

    if manifest is None:
        raise RuntimeError(
            "OBD result tables exist but obd_run_manifest.json is missing. "
            "Rerun the PRE-PAPER-FINAL-R6-DRFIX OBD runner before analysis."
        )

    expected_manifest = {
        "runner_revision": EXPECTED_OBD_RUNNER_REVISION,
        "metrics_version": EXPECTED_OBD_METRICS_VERSION,
        "oracle_metrics_present": False,
        "ambiguity_aware_reporting": True,
        "action_context_alignment": "item_id_plus_bijective_categorical_relabeling",
    }
    for key, expected in expected_manifest.items():
        got = manifest.get(key)
        if got != expected:
            raise RuntimeError(
                "OBD manifest is not compatible with R6 analysis: "
                f"{key}={got!r}, expected {expected!r}."
            )

    required = {
        "obd_decision_results.csv": (
            obdd,
            {
                "campaign",
                "direction",
                "method",
                "deployment_coverage",
                "clear_harmful_exposure",
                "ambiguous_deployment_exposure",
                "clear_beneficial_deployment_exposure",
                "clear_sign_agreement",
                "reference_clear_policy_value",
                "hybrid_minus_reference_clear_policy",
                "reference_sign_selective_value",
                "hybrid_minus_reference_sign_selective",
                "clear_reference_weight",
                "ambiguous_reference_weight",
                "n_clear_groups",
                "n_ambiguous_groups",
                "metrics_version",
                "runner_revision",
            },
        ),
        "obd_h8_evidence_status.csv": (
            obd_h8,
            {
                "campaign",
                "direction",
                "h8_state",
                "n_clear_groups",
                "n_ambiguous_groups",
                "clear_reference_weight",
                "ambiguous_reference_weight",
                "interpretation",
                "runner_revision",
            },
        ),
        "obd_reference_precision_summary.csv": (
            obd_ref_precision,
            {
                "campaign",
                "direction",
                "n_clear_groups",
                "n_ambiguous_groups",
                "clear_reference_weight",
                "ambiguous_reference_weight",
                "h8_reference_testability",
                "runner_revision",
            },
        ),
        "obd_evaluability_summary.csv": (
            obd_evaluability,
            {
                "campaign",
                "direction",
                "mean_local_ess_fraction",
                "median_local_w_q99",
                "max_local_w_max",
                "min_local_pscore",
                "runner_revision",
            },
        ),
    }

    for name, (df, cols) in required.items():
        if df is None:
            raise RuntimeError(
                f"R6 OBD analysis requires {name}, but it is missing. "
                "Rerun run_obdnew.py with PRE-PAPER-FINAL-R6-DRFIX."
            )
        _require_columns(df, name, cols)
        bad = sorted(FORBIDDEN_OBD_ORACLE_COLUMNS & set(df.columns))
        if bad:
            raise RuntimeError(
                f"R6 OBD analysis rejects obsolete oracle columns in {name}: {bad}."
            )
        if "runner_revision" in df.columns:
            revisions = set(df["runner_revision"].dropna().astype(str).unique())
            if revisions != {EXPECTED_OBD_RUNNER_REVISION}:
                raise RuntimeError(
                    f"{name} contains runner revisions {sorted(revisions)}, "
                    f"expected only {EXPECTED_OBD_RUNNER_REVISION}."
                )

    versions = set(obdd["metrics_version"].dropna().astype(str).unique())
    if versions != {EXPECTED_OBD_METRICS_VERSION}:
        raise RuntimeError(
            "obd_decision_results.csv has incompatible metrics_version values: "
            f"{sorted(versions)}."
        )


def _full_or_latest_obd_rows(q: pd.DataFrame) -> pd.DataFrame:
    """Prefer the full-log decision row; otherwise retain the largest available n."""
    if q.empty:
        return q
    if "n_label" in q.columns:
        full = q[q["n_label"].astype(str).eq("full")]
        if len(full):
            return full
    if "n" in q.columns and q["n"].notna().any():
        return q[q["n"] == q["n"].max()]
    return q


def _analyze_h8_r6(rows, obdd, obd_h8, obd_ref_precision, obd_evaluability, manifest):
    """R6 ambiguity-aware H8 analysis; never invents an OBD oracle regret."""
    hypothesis = "H8 OBD independent on-policy validation"

    if obdd is None and obd_h8 is None and obd_ref_precision is None:
        _add(
            rows,
            hypothesis,
            "pending",
            "OBD output",
            "not run",
            "Formal article support requires the OBD run; bundled 10k data are smoke-test only.",
        )
        return

    run_mode = str((manifest or {}).get("run_mode", "unknown"))
    data_path = str((manifest or {}).get("data_path", "unknown"))

    # R6 makes the runner-generated H8 status table authoritative for whether
    # the independent reference is precise enough to score local decisions.
    for _, st in obd_h8.iterrows():
        campaign = str(st["campaign"])
        direction = str(st["direction"])
        state = str(st["h8_state"])
        n_clear = int(st["n_clear_groups"])
        n_amb = int(st["n_ambiguous_groups"])
        clear_w = float(st["clear_reference_weight"])
        amb_w = float(st["ambiguous_reference_weight"])

        tag = f"{campaign}/{direction}"
        detail = (
            f"clear_groups={n_clear}, ambiguous_groups={n_amb}, "
            f"clear_reference_weight={clear_w:.6g}, "
            f"ambiguous_reference_weight={amb_w:.6g}; "
            f"run_mode={run_mode}, data={data_path}."
        )

        if state == "pending_reference_precision" or n_clear == 0:
            _add(
                rows,
                hypothesis,
                "pending_reference_precision",
                f"{tag} H8 reference testability",
                state,
                "R6: no clear local on-policy reference groups exist, so local sign agreement "
                "and clear-harm exposure MUST NOT be scored as evidence of safety. " + detail,
            )
            continue

        if state not in {"evaluable_on_clear_groups", "testable_on_clear_groups"}:
            _add(
                rows,
                hypothesis,
                "warning",
                f"{tag} H8 reference testability",
                state,
                "Unexpected R6 H8 state; inspect obd_h8_evidence_status.csv before making a claim. "
                + detail,
            )
            continue

        q = obdd[
            (obdd["campaign"].astype(str) == campaign)
            & (obdd["direction"].astype(str) == direction)
            & (obdd["method"].astype(str) == "bb_dr")
        ].copy()
        q = _full_or_latest_obd_rows(q)
        if q.empty:
            _add(
                rows,
                hypothesis,
                "warning",
                f"{tag} R6 decision rows",
                "missing",
                "H8 is reference-testable but no matching bb_dr decision row was found.",
            )
            continue

        sign_vals = pd.to_numeric(q["clear_sign_agreement"], errors="coerce").dropna()
        if sign_vals.empty:
            _add(
                rows,
                hypothesis,
                "warning",
                f"{tag} clear-group Gate/reference agreement",
                np.nan,
                "R6 says clear groups exist, but clear_sign_agreement is unavailable; inspect the OBD decision output.",
            )
        else:
            sign = float(sign_vals.mean())
            # Preserve the old pre-specified descriptive threshold while scoring
            # ONLY clear groups. Smoke runs remain diagnostic, not article evidence.
            if run_mode == "paper":
                status = "support" if sign >= 0.70 else "warning"
            else:
                status = "smoke-diagnostic"
            _add(
                rows,
                hypothesis,
                status,
                f"{tag} clear-group Gate/reference agreement",
                sign,
                "Agreement is computed only on R6 clear beneficial/harmful reference groups; "
                "ambiguous groups are excluded. " + detail,
            )

        # R6 value diagnostics are comparisons to independent on-policy reference
        # deployment rules. They are NOT named or interpreted as oracle regret.
        for col, metric, meaning in [
            (
                "hybrid_minus_reference_clear_policy",
                "hybrid minus reference-clear policy value",
                "Difference between the OPE-gated hybrid value and the policy that follows only statistically clear reference signs.",
            ),
            (
                "hybrid_minus_reference_sign_selective",
                "hybrid minus reference-sign-selective value",
                "Difference between the OPE-gated hybrid value and the reference sign-selective benchmark; this is a noisy reference comparison, not oracle regret.",
            ),
        ]:
            vals = pd.to_numeric(q[col], errors="coerce").dropna()
            if len(vals):
                _add(
                    rows,
                    hypothesis,
                    "descriptive",
                    f"{tag} {metric}",
                    float(vals.mean()),
                    meaning + " R6 on-policy references remain noisy and non-oracular.",
                )

    # Separate evidence-quality diagnostics: report them without turning them
    # into a safety claim.
    if obd_ref_precision is not None and len(obd_ref_precision):
        for _, r in obd_ref_precision.iterrows():
            tag = f"{r['campaign']}/{r['direction']}"
            _add(
                rows,
                hypothesis,
                "descriptive",
                f"{tag} ambiguous reference weight",
                float(r["ambiguous_reference_weight"]),
                "R6 reference precision diagnostic; high ambiguity limits H8 identifiability.",
            )

    if obd_evaluability is not None and len(obd_evaluability):
        for _, r in obd_evaluability.iterrows():
            tag = f"{r['campaign']}/{r['direction']}"
            _add(
                rows,
                hypothesis,
                "descriptive",
                f"{tag} mean local ESS fraction",
                float(r["mean_local_ess_fraction"]),
                "R6 policy-coverage/evaluability diagnostic; this concerns OPE evidence quality, not on-policy reference precision.",
            )


def analyze_results():
    cfg.print_protocol()
    print("\n[ANALYZE] Article-level evidence summary -- R6 compatible\n")
    rows = []

    sg = _read("synthetic_gate_results_long.csv")
    sd = _read("synthetic_decision_results.csv")
    sp = _read("synthetic_point_estimates_long.csv")
    semg = _read("semi_gate_results_long.csv")
    semd = _read("semi_decision_results.csv")

    obdref = _read("obd_onpolicy_reference.csv")
    obdd = _read("obd_decision_results.csv")
    obd_h8 = _read("obd_h8_evidence_status.csv")
    obd_ref_precision = _read("obd_reference_precision_summary.csv")
    obd_evaluability = _read("obd_evaluability_summary.csv")
    obd_manifest = _read_json(cfg.RESULTS_DIR / "obd_run_manifest.json")

    _validate_r6_obd_outputs(
        obdd,
        obd_h8,
        obd_ref_precision,
        obd_evaluability,
        obd_manifest,
    )

    # H1: global positive can hide local harm.
    if sg is not None:
        q = sg[(sg.method == "bb_dr") & (~sg.scenario.eq("S6_temporal"))].copy()
        keys = [
            c
            for c in ["scenario", "level", "rep", "seed", "n", "nuisance_model"]
            if c in q.columns
        ]
        cnt = tot = 0
        for _, d in q.groupby(keys, dropna=False):
            ov = d[d.group == -1]
            loc = d[d.group >= 0]
            if len(ov) and len(loc):
                tot += 1
                cnt += int(
                    float(ov.delta_true.iloc[0]) > 0
                    and float(loc.delta_true.min()) < 0
                )
        val = cnt / max(tot, 1)
        _add(
            rows,
            "H1 global != local safety",
            "support" if cnt else "not-observed",
            "fraction global-positive with local-negative",
            val,
            "A positive global improvement can coexist with a harmful subgroup."
            if cnt
            else "No sign-reversal observed in completed runs.",
        )

    # H2: evidence quality should control uncertainty.
    if sg is not None:
        q = sg[
            (sg.scenario == "S1_overlap")
            & (sg.method == "bb_dr")
            & (sg.group >= 0)
        ]
        if len(q) > 2:
            corr = float(q[["ess_fraction", "interval_width"]].corr().iloc[0, 1])
            _add(
                rows,
                "H2 uncertainty tracks evaluability",
                "support" if corr < 0 else "warning",
                "corr(ESS/n, interval width)",
                corr,
                "Expected direction is negative: worse overlap -> wider uncertainty.",
            )

    if semg is not None:
        q = semg[
            (semg.method == "bb_dr")
            & (semg.group >= 0)
            & (semg.group_scheme == "10_20_70")
        ]
        if len(q):
            s = q.groupby("regime").agg(
                ess=("ess_fraction", "mean"), width=("interval_width", "mean")
            )
            if {"good_overlap", "poor_overlap"}.issubset(s.index):
                gap = float(
                    s.loc["poor_overlap", "width"] - s.loc["good_overlap", "width"]
                )
                _add(
                    rows,
                    "H2 uncertainty tracks evaluability",
                    "support" if gap > 0 else "warning",
                    "semi poor-good interval width",
                    gap,
                    "Truth/policies are fixed while logging overlap changes.",
                )

    # H3: selective Gate should lower harmful exposure without zero coverage.
    if sd is not None and "global_candidate_harmful_exposure" in sd.columns:
        q = sd[sd.method == "bb_dr"].copy()
        q["harm_reduction"] = (
            q.global_candidate_harmful_exposure - q.harmful_exposure
        )
        red = float(q.harm_reduction.mean())
        cov = float(q.deployment_coverage.mean())
        _add(
            rows,
            "H3 selective safety-coverage",
            "support" if red > 0 and cov > 0 else "warning",
            "mean harmful-exposure reduction vs global rollout",
            red,
            f"Mean deployment coverage={cov:.3f}; safety is not credited if coverage collapses to zero.",
        )

    # H4: DR robustness under misspecification.
    if sp is not None:
        q = sp[
            (sp.scenario == "S4_misspecified")
            & (sp.level == "strong_nonlinear")
            & (sp.group >= 0)
        ]
        if len(q):
            rm = q.groupby("estimator").sq_error.mean().pow(0.5)
            if "dm" in rm and "dr" in rm:
                diff = float(rm["dm"] - rm["dr"])
                _add(
                    rows,
                    "H4 DR under misspecification",
                    "support" if diff > 0 else "not-supported",
                    "RMSE(DM)-RMSE(DR)",
                    diff,
                    "Positive means DR is more robust than DM in the pre-specified nonlinear stress.",
                )

    # H5: partial pooling rare groups: efficiency must not increase harm.
    if sd is not None:
        q = sd[sd.scenario.astype(str).str.startswith("S2_", na=False)]
        if len(q):
            agg = {
                "harm": ("harmful_exposure", "mean"),
                "cov": ("deployment_coverage", "mean"),
            }
            if "oracle_regret" in q.columns:
                agg["regret"] = ("oracle_regret", "mean")
            s = q.groupby("method").agg(**agg)
            if "hier_eb_dr" in s.index and "bb_dr_independent" in s.index:
                harmdiff = float(
                    s.loc["hier_eb_dr", "harm"]
                    - s.loc["bb_dr_independent", "harm"]
                )
                _add(
                    rows,
                    "H5 rare-group partial pooling",
                    "support" if harmdiff <= 0 else "warning",
                    "hierarchical minus independent harmful exposure",
                    harmdiff,
                    "Partial pooling is retained only if it does not increase harmful deployment.",
                )

    # H6: shift-aware extension.
    if sd is not None:
        q = sd[sd.scenario == "S3_shift"]
        if len(q):
            s = q.groupby("method").harmful_exposure.mean()
            if "bb_dr_shift_estimated" in s.index and "bb_dr" in s.index:
                diff = float(s["bb_dr"] - s["bb_dr_shift_estimated"])
                _add(
                    rows,
                    "H6 external-validity shift",
                    "support" if diff >= 0 else "warning",
                    "standard minus estimated-shift harmful exposure",
                    diff,
                    "Positive means estimated shift weighting reduced deployment harm.",
                )

    # H7: natural and local-stress play different roles.
    if semd is not None:
        q = semd[
            (semd.method == "bb_dr") & (semd.group_scheme == "10_20_70")
        ]
        if len(q):
            s = q.groupby("candidate_variant").agg(
                harm=("harmful_exposure", "mean"),
                cov=("deployment_coverage", "mean"),
            )
            _add(
                rows,
                "H7 semi natural vs controlled stress",
                "descriptive",
                "available candidate variants",
                ",".join(map(str, s.index.tolist())),
                "Natural evidence and deliberately local-stressed evidence are reported separately, not pooled into one claim.",
            )

    # H8: PRE-PAPER-FINAL-R6-DRFIX independent on-policy reference analysis.
    _analyze_h8_r6(
        rows,
        obdd,
        obd_h8,
        obd_ref_precision,
        obd_evaluability,
        obd_manifest,
    )

    # H9: historical relevance vs precision.
    if sg is not None:
        q = sg[
            (sg.scenario == "S6_temporal")
            & (sg.method == "bb_dr")
            & (sg.group >= 0)
        ].copy()
        if len(q):
            q["abs_error"] = (q.estimate_draw_mean - q.delta_true).abs()
            s = q.groupby(["level", "relevance_scheme"]).agg(
                error=("abs_error", "mean"),
                ess=("weighted_ess", "mean"),
                width=("interval_width", "mean"),
            )
            improvements = []
            precision_cost = []
            for _, d in s.groupby(level=0):
                dd = d.droplevel(0)
                if "full" not in dd.index:
                    continue
                for scheme, row in dd.iterrows():
                    if scheme == "full":
                        continue
                    improvements.append(float(dd.loc["full", "error"] - row.error))
                    precision_cost.append(float(dd.loc["full", "ess"] - row.ess))
            if improvements:
                _add(
                    rows,
                    "H9 historical relevance vs precision",
                    "descriptive",
                    "mean error improvement from forgetting",
                    float(np.mean(improvements)),
                    f"Mean weighted-ESS reduction={np.mean(precision_cost):.3f}. "
                    "Positive error improvement with positive ESS loss is the intended relevance/precision trade-off, not proof that 'recent is always better'.",
                )

    report = pd.DataFrame(rows)
    save_csv(report, cfg.TABLES_DIR / "article_hypothesis_evidence.csv")
    print_frame("BC-SPI article hypothesis evidence", report, 100)

    txt = cfg.RESULTS_DIR / "article_hypothesis_evidence.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    with open(txt, "w", encoding="utf-8") as f:
        f.write("BC-SPI ARTICLE-LEVEL EVIDENCE SUMMARY -- R6 COMPATIBLE\n\n")
        f.write(report.to_string(index=False))
        f.write(
            "\n\nInterpretation discipline:\n"
            "- 'support' means the completed experiment moved in the pre-specified direction; it is not a proof.\n"
            "- 'pending' means the required data layer has not been run.\n"
            "- 'pending_reference_precision' is the R6 OBD state in which independent on-policy local references are too ambiguous to score H8.\n"
            "- OBD clear-harm exposure or sign agreement is never interpreted as safety evidence when reference groups are ambiguous.\n"
            "- OBD independent on-policy references are noisy references, not counterfactual oracles; no OBD oracle-regret metric is used.\n"
        )
        if obd_manifest is not None:
            f.write("\nOBD R6 manifest:\n")
            f.write(json.dumps(obd_manifest, indent=2, ensure_ascii=False))
            f.write("\n")

    print(f"[saved] {txt}")
    return report


if __name__ == "__main__":
    analyze_results()
