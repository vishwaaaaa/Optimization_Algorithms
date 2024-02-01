"""
This is a template for Assignment 3: unconstrained optimization

You can (and should) call other fctions or import fctions from other files,
but make sure you do not change the fction signature (i.e., fction name `uncon_optimizer`, inputs, and outputs) in this file.
The autograder will import `uncon_optimizer` from this file. If you change the fction signature, the autograder will fail.
"""

import numpy as np


def BFGS(x0, f,g, tau=10**(-5), a_init=1):
    """
    Inputs- 
    x0 : starting point
    tau : convergence tolerance 

    Outputs-
    xstar : optimal point
    fstar : Minimum fction value
    """
    k= 0
    #a_init = 1
    a = a_init
    reset = True
    # Keeping track of the history
    h_x=[x0]
    h_f = []
    while abs(np.max(g(x0)))> tau:
        print("infi norm{}".format(np.max(g(x0))))
        if (k==0 or reset == True) :
            # a=a_init 
            V = (1/(np.linalg.norm(g(x0))))*np.identity(np.size(x0))
        else:
            a = a*abs((np.dot(np.transpose(g(x0)),pold))/(np.dot(np.transpose(g(x0+ a*pold)),p)))
            s = x0 - xold
            y = g(x0) - g(xold)

            sigma_a = 1/(np.dot(np.transpose(s),y))
            
            V = (np.identity(np.size(x0)) - sigma_a*np.outer(s,np.transpose(y))) @ V @ (np.identity(np.size(x0)) - sigma_a*np.outer(y,np.transpose(s))) + sigma_a*(np.outer(s, np.transpose(s)))

        p= -np.dot(V,g(x0)) 
        
        pold = p
        # a,h_ = bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold)
        a,h_ = Backtrack(a,pold,x0)
        xold = x0
        x0 = x0 + a*p
        k=k+1
        if (np.linalg.norm(phi_(0,x0,p)) > 0.7*np.linalg.norm(g(x0))): #0.7
            print("reset done!")
            reset = True
        else :
            reset = False
        print("the Major iteration {} and the optimal value {} and \ng {}".format(k,x0,g(x0)))
        print("the optimal at the line search {}".format(a))
        print("infinity norm.{}".format(max(g(x0))))

        h_x.append(x0)
        h_f.append(f(x0))
    return x0, f(x0),  np.array(h_x), np.array(h_f), np.array(h_)


## Bracktracking line search
def Backtrack(a_init, p, x0,f,g,  mu_1=10**(-6), rho= 0.85):  #0.601, for n=6 Rosenbrock 0.603 mu_1= 10**(-4) n=8 -1,-1 0.3017, for n=6 BFGS 0.8 is good
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
    while (phi(a,x0,p,f) > phi(0,x0,p,f) + mu_1*a*phi_(0,x0,p,g)):
        #print("phi(a){} and phi(0) {}".format(phi(a,x0,p),phi(a,x0,p)))
        a = a* rho
        k=k+1
        
        # print("line search iteration {} and a {}".format(k,a))
        # print("phi {} and phi0 {} and extra {}".format(phi(a,x0,p),phi(0,x0,p), mu_1*a*phi_(0,x0,p)))
        h_a.append(a)
        
        
    h_a = np.array(h_a)
    return a, h_a

## Bracketing phase for the line search 
def bracketing(x, a_init, Phi, Phi_,p,func, mu_1=10**(-5), mu_2 = 0.7, sigma= 2): #bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold)  ## n=6 mu2 = 0.39, sigma = 2.0201
    """
    Input -
    x : current position 
    a_init : the initial proposed step
    phi, phi_ : fction of alpha and the gradient respectively
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
        
        Phi_2 = phi(a_2,x ,p,func)
        h_a.append(a_2)
        if (Phi_2 >Phi_1 + mu_1*a_2*Phi_1_) or (~first and Phi_2 > Phi_1 ):
            Phi_2_ = phi_(a_2,x,p,func)
            
            a_star = pinpoint(x,a_1, a_2, Phi, Phi_, Phi_1, Phi_2, Phi_1_, Phi_2_,p,func)
            # return a_star, np.array(h_a.append(a_star))
            h_a.append(a_star)
            return a_star, h_a
        Phi_2_ = phi_(a_2,x,p,func)
        if (abs(Phi_2_) <= -mu_2*Phi_): #-mu_2*Phi_
            a_star = a_2
            h_a.append(a_star)
            return a_star, h_a
        elif Phi_2_ >= 0:
            a_star = pinpoint(x,a_2,a_1,Phi, Phi_, Phi_2, Phi_1,Phi_2_, Phi_1_,p,func)
            h_a.append(a_star)
            return a_star, h_a
            # return a_star, np.array(h_a.append(a_star))
        else:
            a_1 = a_2
            a_2 = sigma*a_2

        first = False

##### pinpointing
def pinpoint(x,a_low,a_high,Phi_0, Phi_0_, Phi_low, Phi_high, Phi_low_, Phi_high_,p,func, mu_1 = 10**(-5), mu_2=0.7): #NLCG diff n=2 0.419, 0.4199 0.4194
    """
    Inputs -
    a_low : interval endpoint with lower fction value
    a_high : interval endpoint with higher fction value
    Phi_0, Phi_low, Phi_high, Phi_0_ : the alpha fction values and gradients
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
        Phi_p = phi(a_p,x ,p,func)
        
        if Phi_p > Phi_0 + mu_1*a_p*Phi_0_ or Phi_p > Phi_low :
            # print("new high reached {}".format(a_p))
            a_high = a_p
            Phi_high = Phi_p # for the quadratic interpolation
        else:
            Phi_p_ = phi_(a_p,x,p,func)
            
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
    # b = np.array([Phi_low, Phi_high, Phi_low_])
    # A = np.array([[1, a_low, a_low**2],[1,a_high, a_high**2], [0, 1, 2*a_low]])
    # if (np.linalg.det(A)==0):
    #     print(A)
    # c = np.linalg.solve(A,b)
    # a_star = -c[1]/(2*c[2])
    a_star = (2*a_low *(Phi_high - Phi_low) + Phi_low_*(a_low**2 - a_high**2))/(2*(Phi_high-Phi_low + Phi_low_*(a_low - a_high)))
    return a_star
    

### fctions and Gradients computations ###
def phi(a, x, p,func):
    """
    The fction of alpha
    Input- 
    a : the alpha
    x : current position
    p : the direction of step

    Output-
    fction of alpha
    """
    f,g = func(x+a*p)
    return f

def phi_(a,x,p,func):
    """
    The derivative of the fction with alpha
    Input-
    a : the alpha
    x : current position
    p : the direction of step

    Output-
    Gradient of line search fction
    """
    f,g = func(x+a*p)
    return np.dot(np.transpose(g),p)




def uncon_optimizer(func, x0, epsilon_g, options=None):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    fc : fction handle
        fction handle to a fction of the form: f, g = fc(x)
        where f is the fction value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        Starting point
    epsilon_g : float
        Convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    options : dict
        A dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in on autograder,
        so if the input is None you should setup some defaults.

    Returns
    -------
    xopt : ndarray
        The optimal solution
    fopt : float
        The corresponding optimal fction value
    output : dictionary
        Other miscelaneous outputs that you might want, for example an array
        containing a convergence metric at each iteration.

        `output` must includes the alias, which will be used for mini-competition for extra credit.
        Do not use your real name or uniqname as an alias.
        This alias will be used to show the top-performing optimizers *anonymously*.
    """

    # TODO: set your alias for mini-competition here
    output = {}
    output['alias'] = 'PLEASE SET YOUR ALIAS HERE'

    if options is None:
        # TODO: set default options here.
        # You can pass any options from your subproblem runscripts, but the autograder will not pass any options.
        # Therefore, you should sse the  defaults here for how you want me to run it on the autograder.
        tau = 10**(-5)
        a_init = 1


    # TODO: Your code goes here!
   
    f,g =func(x0)
    
    """
    Inputs- 
    x0 : starting point
    tau : convergence tolerance 

    Outputs-
    xstar : optimal point
    fstar : Minimum ftion value
    """
    k= 0
    #a_init = 1
    a = a_init
    reset = True
    # Keeping track of the history
    h_x=[x0]
    h_f = []
    while max(abs(g))> tau:
        _,g = func(x0)
        # print("infi norm{}".format(np.max(g)))
        if (k==0 or reset == True) :
            # a=a_init 
            V = (1/(np.linalg.norm(g)))*np.identity(np.size(x0))
        else:
            # a = a*abs((np.dot(np.transpose(g),pold))/(np.dot(np.transpose(g(x0+ a*pold)),p)))
            s = x0 - xold
            _,gold = func(xold)
            y = g - gold

            sigma_a = 1/(np.dot(np.transpose(s),y))
            
            V = (np.identity(np.size(x0)) - sigma_a*np.outer(s,np.transpose(y))) @ V @ (np.identity(np.size(x0)) - sigma_a*np.outer(y,np.transpose(s))) + sigma_a*(np.outer(s, np.transpose(s)))

        p= -np.dot(V,g) 
        
        pold = p
        a,h_ = bracketing(x0, a, phi(0,x0,pold,func), phi_(0,x0,pold,func),pold,func)
        # a,h_ = Backtrack(a,pold,x0,func)
        xold = x0
        x0 = x0 + a*p
        k=k+1
        # if (np.linalg.norm(phi_(0,x0,p,func)) > 0.82*np.linalg.norm(g)): #0.7
        # if (phi_(0,x0,p,func)<0.01):
        #     #print("reset done!")
        #     reset = True
        # else :
        reset = False
        # print("the Major iteration {} and the optimal value {} and \ng {}".format(k,x0,g))
        # print("the optimal at the line search {}".format(a))
        # print("infinity norm.{}".format(max(g)))

        h_x.append(x0)
        # h_f.append(f(x0))
        xopt = x0
        fopt,_ = func(x0)
        _,g = func(x0)
    # return x0, fun(x0),  np.array(h_x), np.array(h_f), np.array(h_)


    return xopt, fopt, output



    
