#coding: utf-8

import numpy as np
# import kernel_matrix as km
from .. import Gauss_kernel

def HyperPara(x, x_, y, theta, kernel_type = "GP"):
    if kernel_type == "GP":
        N = len(x)
        M = len(x_)
        K_theta = [[Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], i, j, theta) for j in range(M)] for i in range(N)]
        K_theta_inv = np.linalg.inv(K_theta)
        # dK_dtheta = [[km.Gauss_kernel.Gradient_kernel(x[i][:], x_[j][:], i, j, theta) for j in range(M)] for i in range(N)]
        dK_dtheta = Gauss_kernel.Gradient_K(x, x_, theta)
        
        
        dl_1 = - np.trace(np.dot(dK_dtheta, K_theta_inv), axis1=1, axis2=2)
        dl_2_1 = (np.dot(K_theta_inv,y.reshape(-1,1)))
        dl_2_1 = np.dot(np.dot(dl_2_1.T, dK_dtheta).reshape(3,N), dl_2_1)
        dl_2 = dl_2_1.flatten()

        return dl_1 + dl_2
        