import numpy as np
import matplotlib.pyplot as plt

def function(X,Y):
    """
    The function to be plotted
    """
    return X**4 + 3*X**3 + 3*Y**2 - 6*X*Y - 2*Y
def fun(x):
    """
    The function to be evaluated
    """
    return x[0]**4 + 3*x[0]**3 + 3*x[1]**2 - 6*x[0]*x[1] - 2*x[1]


if __name__=='__main__':
    x = np.linspace(-3,3,200)
    y = np.linspace(-3,3,200)

    Y,X= np.meshgrid(y,x)
    Xa = np.array([-1/4, 1/12])
    Xb= np.array([-1-np.sqrt(3),-2/3-np.sqrt(3)])
    Xc= np.array([-1+np.sqrt(3),-2/3+np.sqrt(3)])
    # plt.contour(X,Y,function(X,Y), 50, cmap='Greys')
    # plt.plot(Xa[0], Xa[1],'r-*', label='point A')
    # plt.plot(Xb[0], Xb[1],'g-*', label='point B')
    # plt.plot(Xc[0], Xc[1],'b-*', label='point C')
    
    print("Values at a {}, b{} and c{}".format(fun(Xa),fun(Xb),fun(Xc)))
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend()
    # plt.show()
