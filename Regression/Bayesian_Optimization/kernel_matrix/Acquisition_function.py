# coding: utf-8

import numpy as np
from scipy.stats import norm
import copy
import kernel_matrix as km

def Expected_Improvement(X, Y, X_add, theta, Opt_type = "Maximize"):
    
    eta = 0.01

    if (Opt_type == "Maximize"):
        tau = np.max(Y)
    elif (Opt_type == "Maximize"):
        tau = np.min(Y)

    mu, v = km.kernel_function.predict_F(X, Y, X_add, theta, "GK", "Diagonal")
    sigma = np.sqrt(v)
    if (sigma < 1e-12):
        sigma = 1e-12
    
    imp = mu - tau - eta
    Z = imp/ sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    
    return ei

def Gradient_Descent(X, Y, theta, alpha = 0.1, Opt_type = "Maximize"):
    n = len(X[0][:])
    p = 1
    np.random.seed(42)
    x = np.random.randn(p, n).T

    # def Adam(g, num_i = 100, alpha = 0.01):
    m = 0
    v = 0
    Beta_1 = 0.9
    Beta_2 = 0.999
    eta = 1e-8
    eta_fin = 1e-6
    i = 0
    num_i = 100
    fin_flag = True
    h = 0.001

    while (i < num_i and fin_flag):
        # g = km.Parameter_Optimization.Gradient_Parameter.HyperPara(x, x, y, theta, "GP")
        g = (Expected_Improvement(X, Y, x+h, theta, "Maximize") - Expected_Improvement(X, Y, x, theta, "Maximize"))/h

        m = Beta_1*m + (1 - Beta_1)*g
        v = Beta_2*v + (1 - Beta_2)*g**2
        m_prd = (m)/(1 - Beta_1)
        v_prd = (v)/(1 - Beta_2)

        if (Opt_type == "Maximize"):
            x = x + alpha*(m_prd/(np.sqrt(v_prd) + eta))
        elif (Opt_type == "Minimize"):
            x = x - alpha*(m_prd/(np.sqrt(v_prd) + eta))
            
        x = copy.deepcopy(np.where(x <= 1e-6, 1e-6, x))

        # print ("x >> ")
        # print (x)
        # print ("g >>")
        # print (g)

        if (np.sqrt(np.sum((g)**2)) < eta_fin):
            fin_flag = False

        i += 1

    return x