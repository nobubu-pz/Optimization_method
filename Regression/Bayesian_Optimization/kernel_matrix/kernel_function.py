# coding: utf-8
import kernel_matrix as km

def hyper_P():
    shita_1, shita_2, shita_3 = 1, 0.4, 0.1
    return [shita_1, shita_2, shita_3]

def Kernel_matrix(x, x_, N, M, kernel_type = "GK"):
    if (kernel_type == "GK"):
        shita = hyper_P()
        
        K = [[km.Gauss_kernel.Gauss_kernel(x[i][:], x_[j][:], i, j, shita) for j in range(M)] for i in range(N)]
        
        return K

