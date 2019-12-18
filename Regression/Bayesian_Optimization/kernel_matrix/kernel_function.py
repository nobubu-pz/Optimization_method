# coding: utf-8
import numpy as np
import kernel_matrix as km



def Kernel_matrix(x, x_, shita, kernel_type = "GK", matrix_type = "Diagonal"):
    if (kernel_type == "GK"):
        if (matrix_type == "Diagonal"):
            N = len(x)
            M = len(x_)
            K = [[km.Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], i, j, shita) for j in range(M)] for i in range(N)]
            
        
        elif (matrix_type == "Non_Diagonal"):
            N = len(x)
            M = len(x_)
            K = [[km.Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], -1, j, shita) for j in range(M)] for i in range(N)]
            

    return np.array(K)

# def Kernel_vector(x, x_, M, shita, kernel_type = "GK"):
#     if (kernel_type == "GK"):
        
#         K = [km.Gauss_kernel.Gauss_kernel(x[j][:], x_, -1, j, shita) for j in range(M)]
        
#         return np.array(K)