# coding: utf-8
import numpy as np
import kernel_matrix as km
# import Parameter_Optimization as pm
import matplotlib.pyplot as plt
import copy

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
    return np.sin(5*x)

def read_data(n = 1):
    p = 5

    x = np.random.rand(n, p)
    max_r = 0.1
    min_r = -0.1
    y = ((max_r - min_r) * np.random.rand(p) + min_r) + np.sin(5*x)#  + np.sin(10*x[1])

    return x.T, y.reshape(-1, 1)

def test_x(n = 1):
    p = 100
    x = np.linspace(0, 1, p).reshape(n, p)
    # x = np.random.rand(n, p)
    return x.T

def hyper_P(x, y, theta, Opt_type, num_i = 100, alpha = 0.01):

    if (Opt_type == "GD"):
        for i in range(num_i):
            theta = theta + alpha* km.Parameter_Optimization.Gradient_Parameter.HyperPara(x, x, y, theta, "GP")
            theta = copy.deepcopy(np.where(theta <= 1e-6, 1e-6, theta))

            print ("theta >> ")
            print (theta)
    
    elif (Opt_type == "Adam"):
        m = 0
        v = 0
        Beta_1 = 0.9
        Beta_2 = 0.999
        eta = 1e-8
        eta_fin = 1e-6
        i = 0
        fin_flag = True

        while (i < num_i and fin_flag):
            g = km.Parameter_Optimization.Gradient_Parameter.HyperPara(x, x, y, theta, "GP")

            m = Beta_1*m + (1 - Beta_1)*g
            v = Beta_2*v + (1 - Beta_2)*g**2
            m_prd = (m)/(1 - Beta_1)
            v_prd = (v)/(1 - Beta_2)

            theta = theta + alpha*(m_prd/(np.sqrt(v_prd) + eta))
            theta = copy.deepcopy(np.where(theta <= 1e-6, 1e-6, theta))

            print ("theta >> ")
            print (theta)

            if (np.sqrt(np.sum((g)**2)) < eta_fin):
                fin_flag = False

            i += 1

    return theta

if __name__ == "__main__":
    # Bo = Bayesian_optimization()
    # x, z, A = Bo.read_data()

    n = 1
    x, y = read_data(n)
    # theta = np.array([1.5, 0.7, 0.07])
    np.random.seed(42)
    theta = np.random.randn(3)
    theta = np.where(theta <= 0, -theta, theta)
    # theta = np.array([1.5, 0.7, 0.07])

    x_test = test_x(n)
    # theta = hyper_P(x, y, theta, "GD")
    theta = hyper_P(x, y, theta, "Adam")


    print ("theta >>")
    print (theta)

    K = km.kernel_function.Kernel_matrix(x, x, theta, "GK", "Diagonal")
    K_inv = np.linalg.inv(K)

    k_star = km.kernel_function.Kernel_matrix(x, x_test, theta, "GK", "Non_Diagonal")
    k_starstar = km.kernel_function.Kernel_matrix(x_test, x_test, theta, "GK", "Diagonal")

    mu = k_star.T @ K_inv @ y
    var = k_starstar - k_star.T @ K_inv @ k_star
    v = np.diag(np.abs(var)).reshape(-1,1)

    plt.figure(figsize=(12,8))
    plt.title('The result')
    plt.fill_between(x_test.flatten(), (mu - np.sqrt(v)).flatten(), (mu + np.sqrt(v)).flatten())
    plt.plot(x_test.flatten(), mu , color='red', label='predicted_mean')
    plt.scatter(x.flatten(), y.flatten(), label='traindata')
    plt.plot(x_test.flatten(), truef(x_test.flatten()), label='true_label', color='purple')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()