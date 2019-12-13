# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
plt.style.use('seaborn-darkgrid')

class Linear_Regression:
  def __init__(self, n = 2, d = 1):
    self.n = n
    self.d = d
  
  def read_data(self):
    p = 100

    _x = np.random.rand(self.n, p)*100
    _xd = np.array([_x[0]**2])
    _xd = np.insert(_xd, len(_xd), _x[1]**2, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[0]**3, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[1]**3, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[0]**4, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[1]**4, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[0]**5, axis = 0)
    _xd = np.insert(_xd, len(_xd), _x[1]**5, axis = 0)
    # _xd = np.insert(_xd, len(_xd), _x[0]**6, axis = 0)
    # _xd = np.insert(_xd, len(_xd), _x[1]**6, axis = 0)
    # _xd = np.insert(_xd, len(_xd), _x[0]**7, axis = 0)
    # _xd = np.insert(_xd, len(_xd), _x[1]**7, axis = 0)
    print (_xd)


    x = np.insert(_x, len(_x), _xd, axis = 0)
    A = np.insert( x, 0, [1]* p, axis = 0).T
    z = np.sin(x[0]/10) + np.cos(x[1]/10)
    return x, z, A
  
  def Plot_G(self, x, y, a):
    
    X, Y = np.meshgrid(np.arange(0, 100, 0.1), np.arange(0, 100, 0.1))
    Z = a[0] + a[1]*X + a[2]*Y  + a[3]*X**2 + a[4]*Y**2  + a[5]*X**3 + a[6]*Y**3 + a[7]*X**4 + a[8]*Y**4 + a[9]*X**5 + a[10]*Y**5

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot(x[0], x[1], y,marker="o",linestyle='None')
    ax.plot_wireframe(X, Y, Z, linewidth=0.3)

    plt.show()

  def main(self):
    
    x, y, A = self.read_data() # データ生成
    a = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, y)) # 方程式計算
    print (a)
    self.Plot_G(x, y, a)

 
if __name__ == "__main__":
  lr = Linear_Regression(d = 7)
  lr.main()


