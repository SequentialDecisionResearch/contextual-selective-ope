"""One-click BC-SPI Spyder entry point.

Recommended: RUN_MODE='smoke' first. This script runs Synthetic -> Semi-Synthetic
-> OBD -> article-level analysis. In smoke mode, a missing OBP installation is
reported and the first two layers still finish. In paper mode, OBD failure is
fatal because the real-log layer is required for the paper claim.
"""
from __future__ import annotations
import bcspi_config as cfg
from run_synthetic import run_synthetic
from run_semi_synthetic import run_semi_synthetic
from run_obdnew import run_obd
from analyze_results import analyze_results


def main():
    cfg.print_protocol()
    print("\nSTEP 1/4: SYNTHETIC")
    run_synthetic()
    print("\nSTEP 2/4: SEMI-SYNTHETIC")
    run_semi_synthetic()
    print("\nSTEP 3/4: OPEN BANDIT DATASET")
    try:
        run_obd()
    except Exception as e:
        if cfg.RUN_MODE=="paper":
            raise
        print("\n[SMOKE WARNING] OBD real-data stage was not completed:")
        print(repr(e))
        print("Synthetic and Semi-Synthetic outputs remain valid smoke-test results.")
    print("\nSTEP 4/4: ARTICLE-LEVEL SUMMARY")
    analyze_results()
    print("\nALL AVAILABLE BC-SPI STAGES FINISHED.")
    print(f"Project output root: {cfg.PROJECT_DIR}")


if __name__=="__main__":
    main()
