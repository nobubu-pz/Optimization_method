# coding: utf-8
import numpy as np
import kernel_matrix as km

def predict_F(x, y, x_add, theta, kernel_type = "GK", matrix_type = "Diagonal"):
    K = Kernel_matrix(x, x, theta, "GK", "Diagonal")
    K_inv = np.linalg.inv(K)

    # x_test = test_x(n)

    k_star = Kernel_matrix(x, x_add, theta, "GK", "Non_Diagonal")
    k_starstar = Kernel_matrix(x_add, x_add, theta, "GK", "Diagonal")

    mu = k_star.T @ K_inv @ y
    var = k_starstar - k_star.T @ K_inv @ k_star
    v = np.diag(np.abs(var)).reshape(-1,1)
    
    return mu, v

def Kernel_matrix(x, x_, theta, kernel_type = "GK", matrix_type = "Diagonal"):
    if (kernel_type == "GK"):
        if (matrix_type == "Diagonal"):
            N = len(x)
            M = len(x_)
            K = [[km.Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], i, j, theta) for j in range(M)] for i in range(N)]
            
        
        elif (matrix_type == "Non_Diagonal"):
            N = len(x)
            M = len(x_)
            K = [[km.Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], -1, j, theta) for j in range(M)] for i in range(N)]
            

    return np.array(K)

# def Kernel_vector(x, x_, M, theta, kernel_type = "GK"):
#     if (kernel_type == "GK"):
        
#         K = [km.Gauss_kernel.Gauss_kernel(x[j][:], x_, -1, j, theta) for j in range(M)]
        
#         return np.array(K)