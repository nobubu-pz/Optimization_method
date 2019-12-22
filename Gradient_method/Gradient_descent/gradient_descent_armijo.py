# coding: utf-8
import copy

import random
def F_cal(x):
    return x**2

def gradient_F(x):
    h = 0.01
    return (F_cal(x+h) - F_cal(x-h))/2*h

def Armijo_init_setdata():
    c = random.uniform(0, 1)
    tau = random.uniform(0, 1)
    return c, tau

def Armijo_alpha_update(x, _F, _g_F, ):
    update_flag = True

    while update_flag:
        dk = -1*_g_F
        if (F_cal(x + alpha*dk) > _F + c* alpha* _g_F* (dk)):
            alpha = tau * alpha
            

if __name__ == "__main__":
    x0 = 10
    alpha = 1
    eta = 0.0001
    Loop_flag = True
    c, tau = Armijo_init_setdata()

    while Loop_flag:

        

        x_new = x0 - alpha*gradient_F(x0)
        x0 = copy.deepcopy(x_new)
        print ("x >> ", x0)
        print ("F >> ", F_cal(x0) )
        if (gradient_F(x0)**2 < eta):
            Loop_flag = False

    print ("x >> ", x0)
    print ("F >> ", F_cal(x0) )

