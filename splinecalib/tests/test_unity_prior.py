import os.path as op
import numpy as np
import pandas as pd
import splinecalib as splc
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(splc.__path__[0], 'tests/data_for_tests')


def test_identity_calibration():
    """This tests that the default settings give reasonable performance
    on data that is already well-calibrated.  Specifically, it does not
    deviate too far from the line y=x"""
    npts = 20000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = splc.SplineCalib()
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.02)

def test_identity_calibration_unity():
    """This tests that adding unity prior with high weight to a small, 
    well calibrated data set gives good results.  Specifically, it does not
    deviate too far from the line y=x"""
    npts = 100
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = splc.SplineCalib(unity_prior=True, unity_prior_weight=5000)
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.01)

def test_identity_calibration_reg_param():
    """This tests overriding the choice of reg_param_vec."""
    npts = 5000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = splc.SplineCalib(reg_param_vec=np.logspace(-2,2,41))
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.015)

