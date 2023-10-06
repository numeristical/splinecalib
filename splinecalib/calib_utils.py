"""Calibration of predicted probabilities."""
from __future__ import division

import numpy as np
import scipy as sp
import random
import warnings
from scipy.stats import binom
from loss_fun_c import pen_ll_fun, pen_ll_fun_grad

def _natural_cubic_spline_basis_expansion(z, knots):
    """Does the natural cubis spline bases for a set of points and knots"""
    nr_knots = knots.shape[0]
    nr_data = z.shape[0]

    # we explicitly make z a column vector to facilitate broadcasting, namely in the
    # subtraction z - knots[:nr_knots - 1] -> array shape [nr_data, nr_knots - 1]
    z = z[:, np.newaxis]

    outmat = np.zeros((nr_data, nr_knots), dtype=np.float64)
    outmat[:, 0] = np.ones(nr_data, dtype=np.float64)
    outmat[:, 1] = z.squeeze()

    final_cubic_term = non_negative((z - knots[-1])) ** 3
    numerator = (
        non_negative((z - knots[: nr_knots - 1])) ** 3 - final_cubic_term
    )  # array shape (nr_data, nr_knots-1)
    denominator = knots[-1] - knots[: nr_knots - 1]  # array shape (nr_knots-1, )
    d = numerator / denominator  # array shape (nr_data, nr_knots-1)
    phi = d[:, :-1] - d[:, [-1]]  # n.b. the [] around -1 req'd to retain the dimension

    outmat[:, 2:] = phi
    return outmat


def logreg_cv(X, y, num_folds, reg_param_vec, method, max_iter,
              tol, weightvec=None, random_state=42, reg_prec=4, ps_mode='fast'):
    """Routine to find the best fitting penalized Logistic Regression.

    User must provide, the X, y, number of folds, range of `lambda` parameter
    and other specs for the optimization.
    """
    fn_vec = get_stratified_foldnums(y, num_folds, random_state=random_state)
    preds = np.zeros(len(y))
    ll_vec = np.zeros(len(reg_param_vec))
    start_coef_vec = np.zeros(X.shape[1])
    for i,lam_val in enumerate(reg_param_vec):
        num_folds_to_search = 1 if ps_mode=='fast' else num_folds
        for fn in range(num_folds_to_search):
            X_tr = X[fn_vec!=fn,:]
            y_tr = y[fn_vec!=fn]
            X_te = X[fn_vec==fn,:]
            if weightvec is not None:
                weightvec_tr = weightvec[fn_vec!=fn]
                opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                                               start_coef_vec,
                                                (X_tr, y_tr,
                                                 float(lam_val), weightvec_tr),
                                                method=method,
                                                jac=True,
                                                options={"gtol": tol,
                                                 "maxiter": max_iter})
            else:
                opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                                               start_coef_vec,
                                                (X_tr, y_tr,
                                                 float(lam_val)),
                                                method=method,
                                                jac=True,
                                                options={"gtol": tol,
                                                 "maxiter": max_iter})
            coefs = opt_res.x
            if not opt_res.success:
                warnings.warn("Optimization did not converge for lambda={}".format(lam_val))
            preds[fn_vec==fn] = 1/(1+np.exp(-X_te.dot(coefs)))
        if ps_mode=='fast':
            ll_vec[i]=my_log_loss(y[fn_vec==0],preds[fn_vec==0])
        else:
            ll_vec[i]=my_log_loss(y,preds)
    best_index = np.argmin(np.round(ll_vec,decimals=reg_prec))
    best_lam_val = reg_param_vec[best_index]
    best_loss = ll_vec[best_index]
    if weightvec is not None:
        opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                               start_coef_vec,
                                (X_tr, y_tr,
                                 best_lam_val, weightvec_tr),
                                jac=True,
                                options={"gtol": tol,
                                "maxiter": max_iter})
    else:
        opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                               start_coef_vec,
                                (X_tr, y_tr,
                                 best_lam_val),
                                jac=True,
                                options={"gtol": tol,
                                 "maxiter": max_iter})
    if not opt_res.success:
        warn_str = """Optimization did not converge for final fit.
                    This is usually due to numerical issues.
                    Consider increasing `max_iter` or `tol`"""
        warnings.warn(warn_str)

    return(best_lam_val, ll_vec, opt_res)


def my_logit(vec, base=np.exp(1), eps=1e-16):
    vec = np.clip(vec, eps, 1-eps)
    return (1/np.log(base)) * np.log(vec/(1-vec))


def my_log_loss(truth_vec, pred_vec, eps=1e-16):
    pred_vec = np.clip(pred_vec, eps, 1-eps)
    val = np.mean(truth_vec*np.log(pred_vec)+(1-truth_vec)*np.log(1-pred_vec))
    return(-val)


def get_stratified_foldnums(y, num_folds, random_state=42):
    """Given an outcome vector y, assigns each data point to a fold in a stratified manner.
    
    Assumes that y contains only integers between 0 and num_classes-1
    """
    fn_vec = -1 * np.ones(len(y))
    for y_val in np.unique(y):
        curr_yval_indices = np.where(y==y_val)[0]
        np.random.seed(random_state)
        np.random.shuffle(curr_yval_indices)
        index_indices = np.round((len(curr_yval_indices)/num_folds)*
                                 np.arange(num_folds+1)).astype(int)
        for i in range(num_folds):
            fold_to_assign = i if ((y_val%2)==0) else (num_folds-i-1)
            fn_vec[curr_yval_indices[index_indices[i]:index_indices[i+1]]] = fold_to_assign
    return(fn_vec)

