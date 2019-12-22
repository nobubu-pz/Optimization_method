# coding: utf-8

if __name__ == "__main__":
    x = 20.0
    gen_n = 1000
    h = 0.01

    _F = lambda x: -(x)**2
    _dF = lambda x: (_F(x + h) - _F(x))/h
    _ddF = lambda x: (_dF(x + h) - _dF(x))/h

    for k in range(gen_n):

        x_new = x - ( _dF(x)/ _ddF(x) )
        
        x = x_new
        print ("x >> ", x)
        # # print (_dF(x))
        # # print (_ddF(x))

        # # if (abs(x_new - x) < 1e-10):
        #     # break
        # if (k > 10):
        #     break