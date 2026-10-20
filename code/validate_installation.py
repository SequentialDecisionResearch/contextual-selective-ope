"""Fast pre-flight validation for the BC-SPI code bundle.

Run this FIRST in Spyder. It checks the output directory, core dependencies,
probability/data-contract validation, a tiny generic DR/BB pipeline, and the
OBD action-context mock pipeline. It does not download or require full OBD.
"""
from __future__ import annotations
import sys
import numpy as np

import bcspi_config as cfg
from bcspi_core import BanditData, validate_bandit_data, crossfit_qhat, compute_ope_components, dr_uncertainty_by_group
from run_obdnew import mock_obd_self_test


def main():
    cfg.ensure_directories()
    print("="*78);print("BC-SPI INSTALLATION / CORE PRE-FLIGHT");print("Python:",sys.version);print("Project:", cfg.PROJECT_DIR); print("Output :", cfg.RESULTS_DIR)
    testfile=cfg.RESULTS_DIR/"_write_test.txt";testfile.write_text("BC-SPI write test OK",encoding="utf-8");testfile.unlink()
    print("[PASS] output directory is writable")

    rng=np.random.default_rng(1);n=300;d=4;k=3
    X=rng.normal(size=(n,d));pi0=np.full((n,k),1/k);raw=rng.random((n,k));pi1=raw/raw.sum(axis=1,keepdims=True)
    a=np.array([rng.choice(k,p=pi0[i]) for i in range(n)]);ps=np.full(n,1/k);reward=rng.binomial(1,.5,size=n).astype(float);group=np.repeat(np.arange(3),n//3)
    data=BanditData(X=X,action=a,reward=reward,pscore=ps,pi0=pi0,pi1=pi1,group=group)
    validate_bandit_data(data,"preflight")
    q=crossfit_qhat(data,"linear",n_folds=3,seed=2);comp=compute_ope_components(data,q);draws=dr_uncertainty_by_group(comp.dr_delta,group,50,3,bayesian=True)
    assert np.isfinite(comp.dr_delta).all() and all(np.isfinite(v).all() for v in draws.values())
    print("[PASS] generic cross-fitted qhat + DR + Bayesian bootstrap")
    mock_obd_self_test()
    print("[PASS] all core pre-flight checks completed")
    print("Next: run run_synthetic.py and run_semi_synthetic.py in smoke mode; then run_obdnew.py.")

if __name__=="__main__": main()
