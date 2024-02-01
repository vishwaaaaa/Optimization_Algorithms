import numpy as np
import matplotlib.pyplot as plt


## Steepest descent for deciding the directions and the finding the optimal
def steepest_d(x0, tau=10**(-5), a_init=0.05):
    """
    Inputs-
    x0 : starting point
    tau : convergence tolerance

    Outputs-
    x_star : Optimal point
    f(x_star) : Minimum function value
    """
    k = 0 
    deltaf =1
    # Keeping track of the history
    h_x=[x0]
    h_f = []
    while np.linalg.norm(deltaf) > tau:
        
        # print("Grad. function{}".format(np.linalg.norm(gradf(x0))))
        p = - gradf(x0)/(np.linalg.norm(gradf(x0)))
        # p = np.array([4,0.75]) #for validation

        # Estimating a_init
        if k==0:
            a = a_init
        
        # Uncomment for single variable
        # else :
        #     a = a*(gradf(x0)*pold)/(gradf(x0+ a*pold)*p)

        else :
            a = a*abs((np.dot(np.transpose(gradf(x0)),pold))/(np.dot(np.transpose(gradf(x0+ a*pold)),p)))
    
        pold = p

        # a,h_ = Backtrack(a,pold,x0)
        a,h_ = bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold) ## Phi at a=0
        # print("h inside {}".format(h_))
        
        x0 = x0 + a*pold

        k= k+1

        x = x0
        deltaf = gradf(x0)# - gradf(x0-a*pold)
        print("the Major iteration {} and the optimal value {} and gradf {} and factor {}".format(k,x,gradf(x0), ((np.dot(np.transpose(gradf(x0)),pold))/(np.dot(np.transpose(gradf(x0+ a*pold)),p)))))
        print("the optimal at the line search {}".format(a))

        # stops the iteration after a single line search
        # if k==1:
        #     break
        h_x.append(x)
        h_f.append(fun(x))
    return x, fun(x), np.array(h_x), np.array(h_f), np.array(h_)


## Bracktracking line search
def Backtrack(a_init, p, x0,  mu_1=10**(-4), rho= 0.7): 
    """
    Inputs-
    a_init : initial step length
    mu_1 : the sufficient decrease factor 
    rho : Backtracking factor 

    Outputs -
    a_star : step-size satisfying Stronge Wolf

    """
    a = a_init
    print("Value of a {}".format(a))
    
    
    # finding the search direction
    k = 0
    h_a = [a]
    while (phi(a,x0,p) > phi(0,x0,p) + mu_1*a*phi_(0,x0,p)):
        #print("phi(a){} and phi(0) {}".format(phi(a,x0,p),phi(a,x0,p)))
        a = a* rho
        k=k+1
        
        print("line search iteration {} and a {}".format(k,a))
        print("phi {} and phi0 {} and extra {}".format(phi(a,x0,p),phi(0,x0,p), mu_1*a*phi_(0,x0,p)))
        h_a.append(a)
        
        
    h_a = np.array(h_a)
    return a, h_a

## Bracketing phase for the line search 
def bracketing(x, a_init, Phi, Phi_,p, mu_1=10**(-4), mu_2 = 0.7, sigma= 2): #bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold) 
    """
    Input -
    x : current position 
    a_init : the initial proposed step
    phi, phi_ : function of alpha and the gradient respectively
    mu_1, mu_2 : factor for sufficient decrease and sufficient curvature
    sigma : step size increase factor 

    Output -
    a_star : the acceptable step size
    """
    h_a = []
    a_1 =0
    a_2 = a_init
    Phi_1 = Phi  ### Phi_0
    Phi_1_ = Phi_ ### Phi_0_
    first = True
    while True:
        
        Phi_2 = phi(a_2,x ,p)
        h_a.append(a_2)
        if (Phi_2 >Phi_1 + mu_1*a_2*Phi_1_) or (~first and Phi_2 > Phi_1 ):
            Phi_2_ = phi_(a_2,x,p)
            
            a_star = pinpoint(x,a_1, a_2, Phi, Phi_, Phi_1, Phi_2, Phi_1_, Phi_2_,p)
            # return a_star, np.array(h_a.append(a_star))
            h_a.append(a_star)
            return a_star, h_a
        Phi_2_ = phi_(a_2,x,p)
        if (abs(Phi_2_) <= -mu_2*Phi_): #-mu_2*Phi_
            a_star = a_2
            h_a.append(a_star)
            return a_star, h_a
        elif Phi_2_ >= 0:
            a_star = pinpoint(x,a_2,a_1,Phi, Phi_, Phi_2, Phi_1,Phi_2_, Phi_1_,p)
            h_a.append(a_star)
            return a_star, h_a
            # return a_star, np.array(h_a.append(a_star))
        else:
            a_1 = a_2
            a_2 = sigma*a_2

        first = False

##### pinpointing
def pinpoint(x,a_low,a_high,Phi_0, Phi_0_, Phi_low, Phi_high, Phi_low_, Phi_high_,p, mu_1 = 10**(-4), mu_2=0.7):
    """
    Inputs -
    a_low : interval endpoint with lower function value
    a_high : interval endpoint with higher function value
    Phi_0, Phi_low, Phi_high, Phi_0_ : the alpha function values and gradients
    Phi_low_, Phi_high_ : low (a_1) and high (a_2) gradients

    Outputs -
    a_star : step size statisfying Strong Wolfe conditions
    """
    k = 0 
    while True :
        # a_p = bisection(a_low,a_high)
        # a_p = quadratic(a_low, a_high)
        a_p =quadratic(a_low, a_high, Phi_low,Phi_low_, Phi_high)
        # print("a_p value{}".format(a_p))
        Phi_p = phi(a_p,x ,p)
        
        if Phi_p > Phi_0 + mu_1*a_p*Phi_0_ or Phi_p > Phi_low :
            # print("new high reached {}".format(a_p))
            a_high = a_p
            Phi_high = Phi_p # for the quadratic interpolation
        else:
            Phi_p_ = phi_(a_p,x,p)
            
            if abs(Phi_p_) <= -mu_2*Phi_0_:  ## sufficient curvature for strong Wolfe
                a_star = a_p
                return a_star
            elif Phi_p_*(a_high - a_low) >= 0:
                a_high = a_low
                
            
            a_low = a_p
        k = k+1
        # print("Pinpointing iteration number {}".format(k))

# interpolation

#bisection method
def bisection(a_low, a_high):
    return 0.5*(a_low + a_high)

#Quadratic interpolation
def quadratic(a_low, a_high, Phi_low,Phi_low_, Phi_high):
    b = np.array([Phi_low, Phi_high, Phi_low_])
    A = np.array([[1, a_low, a_low**2],[1,a_high, a_high**2], [0, 1, 2*a_low]])
    c = np.linalg.solve(A,b)
    a_star = -c[1]/(2*c[2])
    return a_star







### Functions and Gradients computations ###
def phi(a, x, p):
    """
    The function of alpha
    Input- 
    a : the alpha
    x : current position
    p : the direction of step

    Output-
    function of alpha
    """
    return fun(x + a*p)

def phi_(a,x,p):
    """
    The derivative of the function with alpha
    Input-
    a : the alpha
    x : current position
    p : the direction of step

    Output-
    Gradient of line search function
    """
    return np.dot(np.transpose(gradf(x + a*p)),p)



def fun(x):
    """
    The function to be optimized
    Input- 
    x : the input values

    Output-
    f : the function values
    """
    #return x**2 + (x-1)**2 + 1
    # return 0.1*x[0]**6 - 1.5*x[0]**4 + 5*x[0]**2 + 0.1*x[1]**4 + 3*x[1]**2 - 9*x[1] + 0.5*x[0]*x[1]
    # return x[0]**2 + x[1]**2
    return (1 -x[0])**2 + 100*(x[1] - x[0]**2)**2 #RosenBrock function
    # return (1 - x[0])**2 + (1 - x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2 #Bean function
    # f=0 # N=6 RosenBrock
    # for i in range(5):
    #     f = f + (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    # return f

def funC(X,Y):
    """
    The function to be optimized
    Input- 
    x : meshgrid values
    y: meshgrid values

    Output-
    f : the function values
    """
    #return x**2 + (x-1)**2 + 1
    # return 0.1*X**6 - 1.5*X**4 + 5*X**2 + 0.1*Y**4 + 3*Y**2 - 9*Y + 0.5*X*Y
    # return x[0]**2 + x[1]**2
    return (1 -X)**2 + 100*(Y - X**2)**2 #Rosenbrock function
    # return (1 - X)**2 + (1 - Y)**2 + 0.5*(2*Y - X**2)**2 #Bean function
    # f=0 # N=6 RosenBrock
    # for i in range(5):
    #     f = f + (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    # return f

def gradf(x):
    """
    The gradient of the function to be optimized
    Input- 
    x : the values

    Ouput -
    df : gradient of function
    """

    # return 2*x + 2*x - 2
    # return np.array([0.6*x[0]**5 - 6*x[0]**3 + 10*x[0]**1 + 0.5*x[1] , 0.4*x[1]**3 + 6*x[1]**1 - 9  + 0.5*x[0]])
    # return np.array([2*x[0],2*x[1]])
    return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] -2, 200*(x[1] - x[0]**2)]) # Rosenvelt derivative

    # return np.array([-2 + 2*x[0] - 4*x[0]*x[1] + 2*x[0]**3, -2 + 6*x[1] - 2*x[0]**2 ]) # bean function
    # df = [] # N=6 RosenBrock
    # for i in range(6):
    #     if (i==5):
    #         df.append(200 * (x[i] - x[i-1]**2))
    #     else:
    #         df.append(-400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i]))
    # return np.array(df)

    
if __name__=="__main__":
    x,f, h_x, h_f,d = steepest_d([-1.25,1.25])
    print("the optimal is {} and minimum is {}".format(x,f))

    # x,f, h_x1, h_f,d = steepest_d([-1.25,-1.25])
    # print("the optimal is {} and minimum is {}".format(x,f))

    # x,f, h_x2, h_f,d = steepest_d([1.75,1.75])
    # print("the optimal is {} and minimum is {}".format(x,f))

    # x,f, h_x3, h_f,d = steepest_d([1.75,-1.75])
    # print("the optimal is {} and minimum is {}".format(x,f))
    # x,f,a,b,c = steepest_d([-1,-1,-1,-1,-1,-1])
    # print("the optimal is {} and minimum is {}".format(x,f))


    # # plotting line search
    # func = []
    # for i in np.linspace(0,1.2,100):
    #     xi = np.array([-1.25,1.25]) + i*np.array([4,0.75])
    #     func.append(fun(xi))

    # plt.plot(np.linspace(0,1.2,100), np.array(func))
    # print("d value{}".format(d))

    # falpha = []
    # for i in d:
    #     xi = np.array([-1.25,1.25]) + i*np.array([4,0.75])
    #     falpha.append(fun(xi))
    # plt.plot(d,np.array(falpha),'*-')
    # plt.show()
    ## plotting contours of optimization functions
    xm = np.linspace(-3,3,200)
    ym = np.linspace(-3,3,200)
    Y,X = np.meshgrid(ym,xm)

    # plt.contour(X, Y, funC(X,Y),60,cmap='Greys')
    # # # plt.plot(np.array(h_x)[0,:],np.array(h_x)[1,:],'*-')
    # # # print("The h value{} and x component {} and y component {}".format(np.array(h_x), np.array(h_x)[:,0], np.array(h_x)[:,1]))
    # plt.plot(np.array(h_x)[:,0],np.array(h_x)[:,1],'r-*')
    # plt.plot(np.array(h_x1)[:,0],np.array(h_x1)[:,1],'b-*')
    # plt.plot(np.array(h_x2)[:,0],np.array(h_x2)[:,1],'g-*')
    # plt.plot(np.array(h_x3)[:,0],np.array(h_x3)[:,1],'y-*')
    # plt.xlabel(r'$x1$')
    # plt.ylabel(r'$x2$')
    # plt.show()
    # c=quadratic(-1, 2, 1,-1, 7)
    # print("The coefficients of the quadratic are{}".format(c))
    # print("Size of df {}".format(np.shape(gradf(np.array([1,1,1,1,1,1])))))
