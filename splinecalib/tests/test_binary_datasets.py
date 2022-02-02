import os.path as op
import numpy as np
import pandas as pd
import splinecalib as splc
import betacal as bc
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(splc.__path__[0], 'tests/data_for_tests')


def test_calibration_mozilla4_1046_rf():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_mozilla4_1046_rf.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_mozilla4_1046_rf.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_mozilla4_1046_cb1():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_mozilla4_1046_cb1.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_mozilla4_1046_cb1.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib*1.01)

def test_calibration_mozilla4_1046_cb2():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_mozilla4_1046_cb2.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_mozilla4_1046_cb2.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_mozilla4_1046_cb3():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_mozilla4_1046_cb3.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_mozilla4_1046_cb3.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_airlines_42493_rf():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_airlines_42493_rf.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_airlines_42493_rf.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_airlines_42493_cb1():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_airlines_42493_cb1.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_airlines_42493_cb1.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_airlines_42493_cb2():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_airlines_42493_cb2.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_airlines_42493_cb2.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_airlines_42493_cb3():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_airlines_42493_cb3.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_airlines_42493_cb3.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_2dplanes_727_rf():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_2dplanes_727_rf.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_2dplanes_727_rf.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_2dplanes_727_cb1():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_2dplanes_727_cb1.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_2dplanes_727_cb1.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib*1.01)

def test_calibration_2dplanes_727_cb2():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_2dplanes_727_cb2.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_2dplanes_727_cb2.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)

def test_calibration_2dplanes_727_cb3():
    """
    We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'calibset_2dplanes_727_cb3.csv'))
    test_set = pd.read_csv(op.join(data_path,'testset_2dplanes_727_cb3.csv'))

    preds_calib_set = calib_set.iloc[:,0].to_numpy()
    y_calib_set = calib_set.iloc[:,1].to_numpy()
    preds_test = test_set.iloc[:,0].to_numpy()
    y_test = test_set.iloc[:,1].to_numpy()
    sc = splc.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)

    bc1 = bc.BetaCalibration()
    bc1.fit(preds_calib_set, y_calib_set)
    bc_preds_test_calibrated = bc1.predict(preds_test)
    bc_ll_calib = log_loss(y_test, bc_preds_test_calibrated)

    assert(ll_calib<bc_ll_calib)




