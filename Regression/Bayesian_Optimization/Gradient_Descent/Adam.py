import numpy as np
import copy

def Adam(g, num_i = 100, alpha = 0.01):

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
        # print ("g >>")
        # print (g)

        if (np.sqrt(np.sum((g)**2)) < eta_fin):
            fin_flag = False

        i += 1
