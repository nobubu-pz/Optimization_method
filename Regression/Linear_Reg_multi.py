# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
plt.style.use('seaborn-darkgrid')

class Linear_Regression:
  def __init__(self, n = 2):
    self.n = n
  
  def read_data(self):
    p = 20

    x = np.random.rand(self.n, p)
    xx = x[0] + x[1]
    A = np.insert( x, 0, [1]* p, axis = 0).T
    z = (-0.5 - 0.5) * np.random.rand(p) + xx
    return x, z, A
  
  def Plot_G(self, x, y, a):
    
    X, Y = np.meshgrid(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001))
    Z = a[0] + a[1]*X + a[2]*Y

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot(x[0], x[1], y,marker="o",linestyle='None')
    ax.plot_wireframe(X, Y, Z)

    plt.show()

  def main(self):
    
    x, y, A = self.read_data() # データ生成
    a = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, y)) # 方程式計算
    self.Plot_G(x, y, a)

 
if __name__ == "__main__":
  lr = Linear_Regression()
  lr.main()


