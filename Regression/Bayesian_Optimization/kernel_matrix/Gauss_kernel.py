# coding: utf-8
import numpy as np

def Gauss_kernel(x, x_, i, j, shita):
    delta = lambda i, j : 1 if i == j else 0
    # print ("x >> ")
    # print (x)
    # print ("x_")
    # print (x_)

    k = shita[0]*np.exp( -np.sum((x - x_)**2)/(shita[1]) ) + shita[2]*delta(i, j)
    # k1 = shita[0]*np.exp( -np.sum((x - x_)**2)/(shita[1]) )
    # print (k1)
    # k2 = shita[2]*delta(i, j)
    # print(k1 )

    return k

