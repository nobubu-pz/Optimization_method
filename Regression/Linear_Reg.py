# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random
plt.style.use('seaborn-darkgrid')

class Linear_Regression:
  def __init__(self, n = 2):
    self.n = n
  
  def read_data(self):
    p = 20
    x = np.random.rand(p)
    A = np.array([np.ones(p), x]).T
    y = (-0.5 - 0.5) * np.random.rand(p) + x
    return x, y, A
  
  def Plot_G(self, x, y, a):
    
    x1 = np.arange(0,1,0.001)
    X1 = np.array([np.ones(len(x1)), x1])
    y1 = np.dot(a,X1)
    plt.scatter(x, y, color = "blue", s = 20, alpha = 0.4, edgecolors="blue")
    plt.scatter(x1, y1, color = "orange", s = 3)
    plt.show()

  def main(self):
    
    x, y, A = self.read_data() # データ生成
    a = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, y)) # 方程式計算
    self.Plot_G(x, y, a)

 
if __name__ == "__main__":
  lr = Linear_Regression()
  lr.main()


