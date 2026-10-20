from pathlib import Path
import os
import sys

PROJECT_DIR = Path(r"C:\study_notes\traval_rec\Contextual_bandit_bayesian_ope_safe_imp\BCSPI_python_code")
RESULTS_DIR = PROJECT_DIR / "bcspi_results"

if not PROJECT_DIR.exists():
    raise FileNotFoundError(f"BC-SPI project directory does not exist: {PROJECT_DIR}")

os.chdir(PROJECT_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Working dir :", os.getcwd())
print("Results dir :", RESULTS_DIR)
print("Python exe  :", sys.executable)

import numpy, pandas, scipy, sklearn, matplotlib, obp
print("packages OK")
