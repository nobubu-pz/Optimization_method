# coding: utf-8

if __name__ == "__main__":
    x = 5.0
    gen_n = 1000
    h = 0.001

    _F = lambda x: x**2 + 1
    _dF = lambda x: (_F(x + h) - _F(x))/h

    for k in range(gen_n):

        x_new = x - ( _F(x)/ _dF(x) )
        
        x = x_new
        print ("x >> ", x)
        if (abs(x_new - x) < 0.0001):
            break
        # if (k > 10):
            # break