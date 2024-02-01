import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x0):
    """
    Evaluates the function of interest at the given x0
    """
    # return (x0[0]-1)**2 + (x0[1]-2)**2
    # return (x0[0]-2)**2 +  x0[1]**2
    global fcount
    fcount+=1
    return (1 - x0[0])**2 + (1-x0[1])**2 + 0.5*(2*x0[1] - x0[0]**2)**2

def s(j,dim,n):
    """
    Inputs- 
    j : the design number or the vertex of the simplex
    dim : component for which the step update is done
    n : the dimensionality of the problem
    Outputs - 
    s : the step length (integer)
    """
    l = 1
  
    if j == dim:
        return (l / (n * np.sqrt(2))) * (np.sqrt(n + 1) - 1) + l / np.sqrt(2)
    else:
        return (l / (n * np.sqrt(2))) * (np.sqrt(n + 1) - 1)
def Neld_Mead(x0,tau_x=10**(-1), tau_f=10**(-6)):
    """
    Inputs - 
    x0 : Starting design (array of dim values)
    tau_x : simplex size tolerances
    tau_f : Function value of s.d tolerances

    Outputs -
    x_star : Optimal point
    """
    #fetch the dimensions
    X = np.array(x0.reshape(-1,1))
    n = np.shape(x0)[0]
    for j in range(1,n+1):
        temp = np.zeros((n))
        for dim in range(n):
        
            temp[dim] = x0[dim] + s(j, dim+1, n)
        X = np.hstack((X, temp.reshape(-1,1)))

    ## Starting the while loop
    delta_x = 1
    delta_f = 1
    count = 0
    while delta_f> tau_f:#delta_x > tau_x and count < 25:
        count +=1
        ### sorting the designs ####
        sorted_X, fun = sort_desi(X)
        # print("sorted_X without xn {}".format(sorted_X[:,:n]))
        xc = np.mean(sorted_X[:,:n],axis=1) #centroid
        xn = sorted_X[:,-1]  # last design after sorting
        
        # print("mean {}".format(xc))
        xr = xc + (xc - xn)    #reflection
        color = (1, 0, 0, count/29)  # RGBA tuple, where the alpha value i/6 controls brightness
        # plt.plot(x, np.sin(x) + i, color=color, label=f'Plot {i}')
        Xplot = np.hstack((X, X[:,0].reshape(-1,1)))
        plt.plot(Xplot[0,:],Xplot[1,:], color=color,linewidth=2)
        # plt.plot(xc[0], xc[1],'ro', label='xc')
        # plt.plot(xn[0], xn[1],'go', label='xn')
        # plt.plot(xr[0], xr[1],'bo', label='xr')
        # plt.legend()
        # plt.show()
        # print("fx0 {}".format(f(sorted_X[:,-2])))
        fxr = f(xr)
        fxn = f(xn)
        if (fxr < fun[0]):
            xe = xc + 2*(xc - xn)
            # print("expansion done!")
            # plt.plot(xe[0], xe[1],'yo', label='xe')
            # plt.close()
            if (f(xe) < fun[0]):
                xn = xe
                sorted_X[:,-1] = xe
            else:
                xn = xr
                sorted_X[:,-1] = xr
        
        elif (fxr <= fun[-2]):
            xn = xr
            sorted_X[:,-1] = xr
        else :
            if fxr > fxn:
                xic = xc - 0.5*(xc - xn)
                if f(xic) < fxn:
                    xn = xic
                    sorted_X[:,-1] = xic
                else :
                    for j in range(1,n+1):
                        sorted_X[:,j] = sorted_X[:,0] + 0.5*(sorted_X[:,j] - sorted_X[:,0])
            else:
                xoc = xc + 0.5*(xc - xn)
                if f(xoc) < fxr :
                    xn = xoc
                    sorted_X[:,-1] = xoc
                else :
                    for j in range(1,n+1):
                        sorted_X[:,j] = sorted_X[:,0] + 0.5*(sorted_X[:,j] - sorted_X[:,0])

        # delta_x = 0
        # delta_x = np.linalg.norm(sorted_X[:,:n] - sorted_X[:,-1],ord= 1)                
        # print("delta_x {}".format(delta_x))
        # delta_x = np.mean(abs(sorted_X[:,:n] - sorted_X[:,-1]))
        # print("delta_x {}".format(delta_x))
        delta_f = abs(f(sorted_X[:,-1])- fun[0])
        X = sorted_X
        # print("count {}".format(count))
    return X,sorted_X, X[:,0]

def sort_desi(X):
    """
    Inputs - 
    X : designs 
    Outputs -
    sorted_X : sorted design based on function value
    """
    # Calculate function values for each column
    function_values = f(X)

    # Get indices that would sort the array based on function values
    indices = np.argsort(function_values)

    # Apply the indices to the array
    sorted_X = X[:, indices]
    return sorted_X, function_values[indices]
if __name__ == "__main__":
    fcount = 0
    x0 = np.array([-1.2,1.5])
    X,sorted_X, best = Neld_Mead(x0)
    # print(sorted_X)
    print("Initial of triangle X{} and the sorted X{}".format(X, sorted_X))
    # plt.scatter(X[0,:],X[1,:])
    print("lengths of the sides of the triangle {} {} {}".format(np.linalg.norm(X[:,0] - X[:,1]),np.linalg.norm(X[:,1] -X[:,2]),np.linalg.norm(X[:,2] -X[:,0])))
    print("optimal {}".format(best))
    print("function evaluations {}".format(fcount))
    result = minimize(f, x0, method='BFGS')

    # opt_path = np.array(opt_path)
    
    # Print the result
    print(result)

    # Set axis limits for better visualization
    # x = np.linspace(-2,3,100)
    # y = np.linspace(-1,3,100)
    # X,Y = np.meshgrid(x,y)
    # plt.contour(X,Y, (1 - X)**2 + (1-Y)**2 + 0.5*(2*Y - X**2)**2,cmap='Greys',levels =80)
    # plt.xlim(-2, 3)
    # plt.ylim(-1, 3)
    # plt.show()