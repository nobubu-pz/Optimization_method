# coding: utf-8
import numpy as np
import kernel_matrix as km


# class Bayesian_optimization:
#     def __init__(self, n = 2):
#         self.n = n
        
#     def read_data(self):
#         p = 20

#         x = np.random.rand(self.n, p)
#         xx = x[0] + x[1]
#         # A = np.insert( x, 0, [1]* p, axis = 0).T
#         y = (-0.5 - 0.5) * np.random.rand(p) + xx

#         return x, y, A


def read_data(n = 2):
    p = 4

    x = np.random.rand(n, p)
    # xx = x[0] + x[1]
    # A = np.insert( x, 0, [1]* p, axis = 0).T
    max_r = 0.1
    min_r = -0.1
    y = ((max_r - min_r) * np.random.rand(p) + min_r) + np.sin(10*x[0]) + np.sin(10*x[1])

    return x.T, y

if __name__ == "__main__":
    # Bo = Bayesian_optimization()
    # x, z, A = Bo.read_data()

    n = 2
    x, y = read_data(n)

    K = km.kernel_function.Kernel_matrix(x, x, len(x), len(x), "GK")

    yy = np.linalg.solve(K, y)

    print (x)
    print (y)
    print (K)
    print (yy)