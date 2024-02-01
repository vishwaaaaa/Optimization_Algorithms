import numpy as np
import matplotlib.pyplot as plt

def Newton1D(u0, e=0.7, M =np.pi/2):
    """
    Input
    u0 : the initial state
    Kep : the residual function
    dKep : the derivative of the residual

    Output
    uf : the approximate solution
    """
    series = []
    diff = []
    delta = 1
    count =  0
    while(abs(delta)>10**(-2)):
        uf = u0 - Kep(u0, e, M)/dKep(u0, e, M)

        delta = uf - u0

        u0 = uf
        count= count +1
        diff.append(delta)
        series.append(uf)

    return uf, count, series, diff

def Kep(u, e=0.7, M = np.pi/2):
    """
    Kepler's function
    """
    # e= 0.7
    # M = np.pi/2
    return u - e*np.sin(u) - M

def dKep(u, e=0.7, M = np.pi/2):
    """
    Kepler's function
    """
    # e= 0.7
    # M = np.pi/2
    return 1 - e*np.cos(u) 

def GS(u0):
    """
    The function for root finding using gauss seidel

    """
    delta = 1
    e= 0.7
    M = np.pi/2
    count= 0
    series = []
    diff = []
    while(abs(delta) > 10**(-2) ):
        uf = e*np.sin(u0) + M
        delta = uf - u0
        u0 = uf
        count= count + 1
        series.append(uf)
        diff.append(delta)

    return uf, count, series, diff


def GSnew(u0,e=0.7,M=np.pi/2):
    """
    The function for root finding using gauss seidel

    """
    delta = 1

    count= 0
    series = []
    diff = []
    while(abs(delta) > 10**(-2) ):
        uf = e*np.sin(u0) + M
        delta = uf - u0
        u0 = uf
        count= count + 1
        series.append(uf)
        diff.append(delta)

    return uf, count, series, diff
if __name__ == "__main__":
    # starting point
    u0 = 10


    # Newton Solver 
    U, iter, S, diff = Newton1D(u0)
    diff = np.array(diff)
    #print(diff)
    print("The root {} and iteration {}".format(U,iter))
    print("The series of solution {} \n Final rate of convergence {}".format(S, np.log10(abs(diff[-2]/diff[-3]))/np.log10(abs(diff[-3]/diff[-4]))))

    # fixed point Solver
    U_, iter_, S_, diffgs = GS(u0)
    diffgs = np.array(diffgs)
    #print(diffgs)
    print("The gauss jacobi root {}, iteration {} and {} \n Final rate of convergence {}".format(U_, iter_, S_, np.log10(abs(diffgs[-2]/diffgs[-3]))/np.log10(abs(diffgs[-3]/diffgs[-4]))))

    # E versus M
    # empty array to 
    E = [] 
    i = 0
    legs = np.array(['e=0','e=0.1','e=0.5','e=0.9'])
    for e in np.array([0, 0.1, 0.5, 0.9]):
        list = []
        
        for M in np.linspace(0,2*np.pi, 16):
            # U, iter, S, diff = Newton1D(u0, e, M)
            U, iter, S, diff = GSnew(u0, e, M)
            list.append(U)
        
        E.append(np.array(list))
        # plt.plot(np.linspace(0,2*np.pi, 16), np.array(E)[i], label=legs[i])
        
        i = i +1
    # print(np.array(E).shape)
    # print(np.shape(np.linspace(0,2*np.pi, 16)))
    # plt.xlabel("M")
    # plt.ylabel("E")
    # plt.legend()
    # plt.show()

    # +++++ numerical error ++++ #
    m = np.linspace(np.pi/2 - 10**(-10), np.pi/2 + 10**(-10), 20 )# np.pi/2 + (10**(-10))*np.random.randn(40)
    

    Mseries = []
    for me in m:
        u0 = 1.0 + 2*np.random.randn(1)
        # U, iter, S, diff = Newton1D(u0, e, me)
        U, iter, S, diff = GSnew(u0, e, me)
        Mseries.append(U)
    
    # plt.plot(m,np.array(Mseries), 'o-')
    # plt.xlabel("M")
    # plt.ylabel("E")
    # plt.show()
    


    





