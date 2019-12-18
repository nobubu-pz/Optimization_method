# coding: utf-8
import numpy as np
import kernel_matrix as km
import matplotlib.pyplot as plt

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

def truef(x):
    # return np.sin(10*x[0][:]) + np.sin(10*x[1])
    return np.sin(10*x[0][:])

def read_data(n = 1):
    p = 4

    x = np.random.rand(n, p)
    max_r = 0.1
    min_r = -0.1
    y = ((max_r - min_r) * np.random.rand(p) + min_r) + np.sin(10*x)#  + np.sin(10*x[1])

    return x.T, y.reshape(-1, 1)

def test_x(n = 1):
    p = 3
    # x = np.linspace(-1, 1, p).reshape(n, p)
    x = np.random.rand(n, p)
    return x.T

def hyper_P():
    shita_1, shita_2, shita_3 = 1, 0.4, 0.1
    return [shita_1, shita_2, shita_3]

if __name__ == "__main__":
    # Bo = Bayesian_optimization()
    # x, z, A = Bo.read_data()

    n = 1
    x, y = read_data(n)
    x_test = test_x(n)
    shita = hyper_P()
    print (x)
    print (y)
    print (x_test)
    K = km.kernel_function.Kernel_matrix(x, x, shita, "GK", "Diagonal")
    K_inv = np.linalg.inv(K)

    k_star = km.kernel_function.Kernel_matrix(x, x_test, shita, "GK", "Non_Diagonal")
    k_starstar = km.kernel_function.Kernel_matrix(x_test, x_test, shita, "GK", "Diagonal")
    
    print (K)
    print (k_star)
    print (k_starstar)

    mu = k_star.T @ K_inv @ y
    var = k_starstar - k_star @ K_inv @ k_star.T

    print (x)
    print (y)
    print (K)
    print (k_star)

    print ("mu >> ")
    print (mu)
    print ("var >> ")
    print (var)

    plt.figure(figsize=(12,8))
    plt.title('The result')
    plt.fill_between(x_test.flatten(), mu -np.sqrt(var), mu +np.sqrt(var))
    plt.plot(x_test.flatten(), mu , color='red', label='predicted_mean')
    plt.scatter(x.flatten(), y.flatten(), label='traindata')
    plt.plot(x_test.flatten(), truef(x_test.flatten()), label='true_label', color='purple')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()