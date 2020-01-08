# coding: utf-8
import numpy as np

delta = lambda i, j : 1 if i == j else 0

def Gauss_kernel(x, x_, i, j, shita):
    # print ("x >> ")
    # print (x)
    # print ("x_")
    # print (x_)

    k = shita[0]*np.exp( -np.sqrt(np.sum((x - x_)**2))/(shita[1]) ) + shita[2]*delta(i, j)
    # k1 = shita[0]*np.exp( -np.sum((x - x_)**2)/(shita[1]) )
    # print (k1)
    # k2 = shita[2]*delta(i, j)
    # print(k1 )

    return k

def Gradient_kernel(x, x_, i, j, shita):
    dK_dtheta = np.zeros(3)
    
    dK_dtheta[0] = np.exp(-(np.sqrt(np.sum((x - x_)**2)))/shita[1])
    dK_dtheta[1] = shita[0] * ((np.sqrt(np.sum((x - x_)**2)))/shita[1]**2) *\
        np.exp(-(np.sqrt(np.sum((x - x_)**2)))/shita[1])
    dK_dtheta[2] = delta(i, j)
    
    return dK_dtheta

def Gradient_K(x, shita):
    N = len(x)
    dK = np.zeros(shape = [N, N, 3])
    for i in range(N):
        for j in range(N):
            dK[i, j] = Gradient_kernel(x[i][:], x[j][:], i, j, shita)
    
    dK = dK.transpose(2, 0, 1)
    return dK


