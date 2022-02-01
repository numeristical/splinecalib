cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport log as clog
from libc.math cimport exp

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def pen_ll_fun(cnp.ndarray[double] beta, 
               cnp.ndarray[double, ndim=2] X, 
               cnp.ndarray[double] y,
               double lam=0, 
               cnp.ndarray[double] weight_vec=None,
               long max_exp=50):
    cdef double ll_term, reg_term
    cdef cnp.ndarray[double] betaX
    cdef long i
    
    betaX = X.dot(beta)
    ll_term = 0
    reg_term = 0
    if weight_vec is None:
        for i in range(betaX.shape[0]):
            if y[i]>0:
                if betaX[i]>= -max_exp:
                    ll_term += y[i]*clog(exp(-betaX[i])+1)
                else:
                    ll_term += y[i]*(-betaX[i])
            if y[i]<1:
                if betaX[i]<= max_exp:
                    ll_term += (1-y[i])*clog(exp(betaX[i])+1)
                else:
                    ll_term +=  (1-y[i])*(betaX[i])
    else:
        for i in range(betaX.shape[0]):
            if y[i]>0:
                if betaX[i]>= -max_exp:
                    ll_term += weight_vec[i]*y[i]*clog(exp(-betaX[i])+1)
                else:
                    ll_term += weight_vec[i]*y[i]*(-betaX[i])
            if y[i]<1:
                if betaX[i]<= max_exp:
                    ll_term += weight_vec[i]*(1-y[i])*clog(exp(betaX[i])+1)
                else:
                    ll_term += weight_vec[i]*(1-y[i])*(betaX[i])
        
    for i in range(beta.shape[0]):
        reg_term+=beta[i]*beta[i]
    reg_term = lam*reg_term/beta.shape[0]
    return(ll_term/X.shape[0]+reg_term)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def pen_ll_fun_grad(cnp.ndarray[double] beta, 
               cnp.ndarray[double, ndim=2] X, 
               cnp.ndarray[double] y,
               double lam=0, 
               cnp.ndarray[double] weight_vec=None,
               long max_exp=50):
    cdef double ll_term, reg_term
    cdef cnp.ndarray[double] betaX, z, main_grad_term, other_grad_term
    cdef long i
    
    betaX = X.dot(beta)
    #expmbx = np.exp(-betaX)
    z = 1/(1+np.exp(-betaX))
    if weight_vec is None:
        main_grad_term = np.mean(X*((-y*(1-z) +(1-y)*z).reshape(-1,1)),axis=0)
    else:
        main_grad_term = np.mean(X*((weight_vec*(-y*(1-z) +(1-y)*z)).reshape(-1,1)),axis=0)
    other_grad_term = beta*2*lam/len(beta)

    
    ll_term = 0
    reg_term = 0
    if weight_vec is None:
        for i in range(betaX.shape[0]):
            if y[i]>0:
                if betaX[i]>= -max_exp:
                    ll_term += y[i]*clog(exp(-betaX[i])+1)
                else:
                    ll_term += y[i]*(-betaX[i])
            if y[i]<1:
                if betaX[i]<= max_exp:
                    ll_term += (1-y[i])*clog(exp(betaX[i])+1)
                else:
                    ll_term +=  (1-y[i])*(betaX[i])
    else:
        for i in range(betaX.shape[0]):
            if y[i]>0:
                if betaX[i]>= -max_exp:
                    ll_term += weight_vec[i]*y[i]*clog(exp(-betaX[i])+1)
                else:
                    ll_term += weight_vec[i]*y[i]*(-betaX[i])
            if y[i]<1:
                if betaX[i]<= max_exp:
                    ll_term += weight_vec[i]*(1-y[i])*clog(exp(betaX[i])+1)
                else:
                    ll_term += weight_vec[i]*(1-y[i])*(betaX[i])
        
    for i in range(beta.shape[0]):
        reg_term+=beta[i]*beta[i]
    reg_term = lam*reg_term/beta.shape[0]
    return((ll_term/X.shape[0]+reg_term), main_grad_term+other_grad_term)