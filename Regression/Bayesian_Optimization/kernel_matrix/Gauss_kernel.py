# coding: utf-8
import numpy as np

delta = lambda i, j : 1 if i == j else 0

def Gauss_kernel(x, x_, i, j, theta):
    # print ("x >> ")
    # print (x)
    # print ("x_")
    # print (x_)

    k = theta[0]*np.exp( -np.sqrt(np.sum((x - x_)**2))/(theta[1]) ) + theta[2]*delta(i, j)
    # k1 = theta[0]*np.exp( -np.sum((x - x_)**2)/(theta[1]) )
    # print (k1)
    # k2 = theta[2]*delta(i, j)
    # print(k1 )

    return k

def Gradient_kernel(x, x_, i, j, theta):
    dK_dtheta = np.zeros(3)
    
    dK_dtheta[0] = np.exp(-(np.sqrt(np.sum((x - x_)**2)))/theta[1])
    dK_dtheta[1] = theta[0] * ((np.sqrt(np.sum((x - x_)**2)))/theta[1]**2) *\
        np.exp(-(np.sqrt(np.sum((x - x_)**2)))/theta[1])
    dK_dtheta[2] = delta(i, j)
    
    return dK_dtheta

def Gradient_K(x, theta):
    N = len(x)
    dK = np.zeros(shape = [N, N, 3])
    for i in range(N):
        for j in range(N):
            dK[i, j] = Gradient_kernel(x[i][:], x[j][:], i, j, theta)
    
    dK = dK.transpose(2, 0, 1)
    return dK


