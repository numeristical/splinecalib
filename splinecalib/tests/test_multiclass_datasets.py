import os.path as op
import numpy as np
import pandas as pd
import splinecalib as splc
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(splc.__path__[0], 'tests/data_for_tests')

def test_mnist_calib():
    """
    This tests a multiclass calibration on data derived from MNIST.
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'mnist4_calib_set.csv'))
    test_set = pd.read_csv(op.join(data_path,'mnist4_test_set.csv'))

    preds_calib_set = calib_set.iloc[:,:-1].to_numpy()
    y_calib_set = calib_set.iloc[:,-1].to_numpy()
    preds_test = test_set.iloc[:,:-1].to_numpy()
    y_test = test_set.iloc[:,-1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)
    assert(ll_calib<.2340)


