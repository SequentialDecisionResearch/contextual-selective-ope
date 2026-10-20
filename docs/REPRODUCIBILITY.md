# Reproducibility and provenance (pre-submission audit)

## Formal-run provenance vs. current runner

- Synthetic: `experiment_protocol/synthetic_run_manifest.json`. Original formal run used seed `20260823`, DGP seed `20260821` in manuscript, 200 regular / 500 key repetitions, `n={2000,5000,10000,20000}`, 1000 ordinary / 1000 Bayesian bootstrap draws, 5 folds, and recovered S0–S5 from 1240 checkpoint shards. **S6 was not present** in this recovered output.
- Semi-Synthetic: `experiment_protocol/semi_run_manifest.json`. Three datasets, 250 primary logging draws, 100 sensitivity draws, 20 outer splits × 10 logging draws.
- OBD: `experiment_protocol/obd_run_manifest.json`. Revision `PRE-PAPER-FINAL-R6-DRFIX`, metrics `obd_eval_v6_drfixed`, campaigns All/Men/Women, `bts_n_sim=100000`, 20 non-full repetitions, static prior BTS evaluation distribution, reference intervals Newcombe–Wilson, cross-fitted Logistic SGD nuisance, and logged on-policy behavior baseline. `temporal_robustness=false`, `oracle_metrics_present=false`.

**Version warning:** `code/bcspi_config.py` in the author's source code was a later `CLEAN FUTURE` protocol with smaller default synthetic scale. Its `paper` mode does **not** recreate the archived full Synthetic run automatically. This bundle makes it portable and defaults to `smoke`.

## Safe quick validation

1. Create a clean Python environment and `pip install -r requirements.txt`.
2. `python code/validate_installation.py` checks core functionality with tiny synthetic data and OBD mock observations. It is **not** a numerical reproduction of paper tables.
3. To run OBD on the original licensed data, set `BCSPI_OBD_DATA_PATH` to the local Open Bandit Dataset installation; full runs can be lengthy and are **not** recommended during this pre-submission audit.

## Tables and figures

Summary snapshots are at `results/paper_snapshot/`; figures in `figures/`. Complete long-form Monte Carlo results and checkpoints were kept separately by the author (large archives). They were not redistributed in this small GitHub package. Contact the authors for archival details; public raw OBD logs are accessed via the original provider.

## Validation limitations

- Large experiments were not rerun during the quick audit.
- The behavior-baseline DR compatibility fix in `code/bcspi_core.py` passed only a mock self-test, not a full independent OBD rerun.
- Several old audits and preliminary summaries remain archived in `results/`; they are not interchangeable with the final manuscript's numbers.
- Credible local lower-bound coverage is an **assumption** of the theoretical false-admission result, not a property guaranteed by bootstrapping in every data regime. Semi-Synthetic empirical nominal 95% coverage can be markedly below 95% under poor overlap.
- The OBD on-policy comparison has ambiguous reference subgroups and does not establish universal real-world non-harm.
