# coding: utf-8
import numpy as np

def _F(X):
    if (X.ndim == 1):
        x1 = X[0]
        x2 = X[1]
        z = x1**2 + x2**2
    elif (X.ndim == 2):
        z = np.zeros(len(X))
        for k in range(len(X)):
            x = X[k][:]
            x1 = x[0]
            x2 = x[1]
            z[k] = x1**2 + x2**2
    return z

if __name__ == "__main__":
    x = np.array([10.0, 12.0])
    gen_n = 1000
    h = 0.0001

    _dF = lambda x, y: (_F(y) - _F(x))/h
    # _ddF = lambda x, y: (_dF(x, y) - _dF(x))/h
    # Hesse = lambda x, y: np.array([_ddF(x, y)])
    Hesse = lambda x, y: _dF(_dF(x, y), y)

    # var_til = np.tile(x, (len(x[0]), 1))
    forw_var = np.tile(x, (len(x), 1)) + np.identity(len(x))*h

    # print (var_til)
    print (forw_var)

    print ("_f(x) >> ")
    print (_F(x))
    print ("_f(x+h) >> ")
    print (_F(forw_var))
    print ("_dfdx >> ")
    print (_dF(x, forw_var))

    print ("Hesse >> ")
    print (Hesse(x, forw_var))
