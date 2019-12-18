# coding: utf-8

import numpy as np

def read_data(self):
    p = 20

    x = np.random.rand(p, self.n)
    xx = x[0] + x[1]
    # A = np.insert( x, 0, [1]* p, axis = 0).T
    y = (-0.5 - 0.5) * np.random.rand(p) + xx

    return x, y


