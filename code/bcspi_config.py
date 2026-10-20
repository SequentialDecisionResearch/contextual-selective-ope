"""BC-SPI frozen research protocol for the faster Windows desktop.

This is the CLEAN FUTURE configuration, not a salvage/recovery script.
The default run mode is PAPER.  Synthetic is intentionally capped at n=10,000
for a better evidence/compute trade-off.  Existing Semi-Synthetic and OBD outputs
remain under the same project result tree.
"""
from __future__ import annotations

from pathlib import Path
import os

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
# Portable reviewer defaults. The original Windows run used an older frozen
# protocol preserved in experiment_protocol/*.json. This configuration is the
# later CLEAN FUTURE runner, not an exact historic snapshot of all paper runs.
PROJECT_DIR = Path(os.environ.get('BCSPI_PROJECT_DIR', str(Path(__file__).resolve().parent.parent))).expanduser().resolve()
RESULTS_DIR = Path(os.environ.get('BCSPI_RESULTS_DIR', str(PROJECT_DIR / 'bcspi_results'))).expanduser().resolve()
if RESULTS_DIR.parent != PROJECT_DIR:
    raise RuntimeError('RESULTS_DIR must be directly under PROJECT_DIR.')
# Safe default prevents accidentally starting a week-long run.
RUN_MODE = os.environ.get('BCSPI_RUN_MODE','smoke').strip().lower()
if RUN_MODE not in {'smoke','paper'}:
    raise ValueError('BCSPI_RUN_MODE must be smoke or paper')

OBD_DATA_PATH: Path | None = (Path(os.environ['BCSPI_OBD_DATA_PATH']).expanduser().resolve() if os.environ.get('BCSPI_OBD_DATA_PATH') else None)
RUN_OBD_TEMPORAL_ROBUSTNESS = False
INCLUDE_COVTYPE = False
COVTYPE_MAX_ROWS = 120_000
SHOW_FIGURES = False

# -----------------------------------------------------------------------------
# COMMON STATISTICAL PROTOCOL
# -----------------------------------------------------------------------------
RANDOM_SEED = 20260823
ALPHA = 0.05
DELTA = 0.0
COVERAGE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
POSTERIOR_GATE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
SAFE_ETA_GRID = [0.00, 0.005, 0.01, 0.02, 0.05]
ESS_SAFEGUARD_FRAC = 0.05
SWITCH_TAU = 10.0
WEIGHT_CLIP_SENSITIVITY = [None, 20.0, 10.0]

# -----------------------------------------------------------------------------
# RUN SCALE
# -----------------------------------------------------------------------------
if RUN_MODE == "smoke":
    N_FOLDS = 3
    BOOTSTRAP_DRAWS = 200
    BAYES_BOOTSTRAP_DRAWS = 250

    SYN_REPS = 3
    SYN_KEY_REPS = 5
    SYN_SAMPLE_SIZES = [2_000]
    SYN_TRUTH_SIZE = 20_000
    SYN_DEPLOY_SIZE = 5_000
    SYN_CHECKPOINT_EVERY = 1
    SYN_S6_SAMPLE_SIZES = [2_000]
    SYN_S6_REPS = 2
    SYN_S6_SCHEME_NAMES = ["full", "decay_h25"]

    SEMI_FIXED_LOG_DRAWS = 3
    SEMI_OUTER_SPLITS = 1
    SEMI_OUTER_LOG_DRAWS = 2
    SEMI_NUISANCE_MODELS = ["linear"]

    OBD_N_FOLDS = 2
    OBD_BTS_N_SIM = 20_000
    OBD_SAMPLE_SIZES = [10_000]
    OBD_REPS_PER_SIZE = 2
    OBD_BOOTSTRAP_DRAWS = 150
    OBD_BAYES_BOOTSTRAP_DRAWS = 200
else:
    N_FOLDS = 5
    BOOTSTRAP_DRAWS = 1_000
    BAYES_BOOTSTRAP_DRAWS = 1_000

    # Compute-efficient PAPER Synthetic protocol.
    # n=20,000 is intentionally omitted: 2k->5k->10k already identifies the
    # finite-sample/ESS/uncertainty trend while avoiding the weakest marginal payoff.
    SYN_REPS = 100
    SYN_KEY_REPS = 200
    SYN_SAMPLE_SIZES = [2_000, 5_000, 10_000]
    SYN_TRUTH_SIZE = 100_000
    SYN_DEPLOY_SIZE = 10_000
    SYN_CHECKPOINT_EVERY = 20

    # S6 is a smaller descriptive robustness layer, not the main evidence engine.
    SYN_S6_SAMPLE_SIZES = [5_000, 10_000]
    SYN_S6_REPS = 60
    SYN_S6_SCHEME_NAMES = ["full", "roll_50", "decay_h25", "relevance_h25"]

    # Keep the existing formal defaults for the other runners.  Their own
    # compute-efficient wrappers may further reduce repeated work.
    SEMI_FIXED_LOG_DRAWS = 500
    SEMI_OUTER_SPLITS = 30
    SEMI_OUTER_LOG_DRAWS = 20
    SEMI_NUISANCE_MODELS = ["linear", "flexible"]

    OBD_N_FOLDS = 2
    OBD_BTS_N_SIM = 100_000
    OBD_SAMPLE_SIZES = [10_000, 50_000, 100_000, 250_000, 500_000, "full"]
    OBD_REPS_PER_SIZE = 50
    OBD_BOOTSTRAP_DRAWS = 500
    OBD_BAYES_BOOTSTRAP_DRAWS = 500

# -----------------------------------------------------------------------------
# SYNTHETIC S6 RECENCY SCHEMES
# -----------------------------------------------------------------------------
RECENCY_SCHEMES = [
    {"name": "full",          "kind": "full",      "param": None},
    {"name": "roll_25",       "kind": "rolling",   "param": 0.25},
    {"name": "roll_50",       "kind": "rolling",   "param": 0.50},
    {"name": "roll_75",       "kind": "rolling",   "param": 0.75},
    {"name": "decay_h10",     "kind": "exp",       "param": 0.10},
    {"name": "decay_h25",     "kind": "exp",       "param": 0.25},
    {"name": "decay_h50",     "kind": "exp",       "param": 0.50},
    {"name": "relevance_h25", "kind": "relevance", "param": 0.25},
]

# -----------------------------------------------------------------------------
# OBD PROTOCOL
# -----------------------------------------------------------------------------
OBD_CAMPAIGNS = ["all", "men", "women"]
OBD_CLUSTER_K = 5
OBD_CLUSTER_DESIGN_MAX_N = 100_000
OBD_REFERENCE_ALPHA = 0.05

# -----------------------------------------------------------------------------
# OUTPUT TREE
# -----------------------------------------------------------------------------
DATA_DIR = PROJECT_DIR / "bcspi_generated_data"
# RESULTS_DIR is fixed explicitly above.
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
LOG_DIR = RESULTS_DIR / "logs"
SYNTHETIC_FIGURES_DIR = FIGURES_DIR / "synthetic"
SEMI_FIGURES_DIR = FIGURES_DIR / "semi_synthetic"
OBD_FIGURES_DIR = FIGURES_DIR / "obd"


def ensure_directories() -> None:
    for d in [
        PROJECT_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
        CACHE_DIR, LOG_DIR, SYNTHETIC_FIGURES_DIR,
        SEMI_FIGURES_DIR, OBD_FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def print_protocol() -> None:
    ensure_directories()
    print("=" * 78)
    print("BC-SPI RESEARCH PROTOCOL")
    print(f"RUN_MODE       : {RUN_MODE}")
    print(f"WORKING_DIR    : {Path.cwd()}")
    print(f"PROJECT_DIR    : {PROJECT_DIR}")
    print(f"RESULTS_DIR    : {RESULTS_DIR}")
    print(f"FIGURES_DIR    : {FIGURES_DIR}")
    print(f"TABLES_DIR     : {TABLES_DIR}")
    print(f"ALPHA / DELTA  : {ALPHA} / {DELTA}")
    print(f"N_FOLDS        : {N_FOLDS}")
    print(f"SYN_SIZES      : {SYN_SAMPLE_SIZES}")
    print(f"SYN_REPS/KEY   : {SYN_REPS} / {SYN_KEY_REPS}")
    print(f"S6_SIZES/REPS  : {SYN_S6_SAMPLE_SIZES} / {SYN_S6_REPS}")
    print(f"SHOW_FIGURES   : {SHOW_FIGURES}")
    print(f"OBD_DATA_PATH  : {OBD_DATA_PATH}")
    print("=" * 78)


ensure_directories()
