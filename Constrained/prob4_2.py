import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

## Steepest descent for deciding the directions and the finding the optimal
def steepest_d(x0,count, tau=10**(-5), a_init=1):
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
    while max(abs(gradf(x0)))> tau:
        
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
            a = a*((np.dot(np.transpose(gradf(xold)),pold))/(np.dot(np.transpose(gradf(x0)),p)))
            # a = a*abs((np.dot(np.transpose(gradf(x0)),pold))/(np.dot(np.transpose(gradf(x0+ a*pold)),p)))
            
        pold = p

        a,h_ = Backtrack(a,pold,x0,count)
        # a,h_ = bracketing(x0, a, phi(0,x0,pold,count), phi_(0,x0,pold),pold,count) ## Phi at a=0
        # print("h inside {}".format(h_))
        xold =x0
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
        # h_f.append(fun(x))
    return x, fun(x,count), np.array(h_x), np.array(h_f), np.array(h_),count


def BFGS(x0, count, tau=10**(-7), a_init=1):
    """
    Inputs- 
    x0 : starting point
    tau : convergence tolerance 

    Outputs-
    xstar : optimal point
    fstar : Minimum function value
    """
    k= 0
    #a_init = 1
    a = a_init
    reset = True
    # Keeping track of the history
    h_x=[x0]
    h_f = []
    gradf_c = [max(abs(gradf(x0)))]
    iter_c = [1]
    while max(abs(gradf(x0)))> tau:
        print("infi norm{}".format(np.max(gradf(x0))))
        if (k==0 or reset == True) :
            # a=a_init 
            V = (1/(np.linalg.norm(gradf(x0))))*np.identity(np.size(x0))
        else:
            # a = a*abs((np.dot(np.transpose(gradf(x0)),pold))/(np.dot(np.transpose(gradf(x0+ a*pold)),p)))
            # a = a*abs((np.dot(np.transpose(gradf(xold)),pold))/(np.dot(np.transpose(gradf(x0)),p)))
            s = x0 - xold
            y = gradf(x0) - gradf(xold)

            sigma_a = 1/(np.dot(np.transpose(s),y))
            
            V = (np.identity(np.size(x0)) - sigma_a*np.outer(s,np.transpose(y))) @ V @ (np.identity(np.size(x0)) - sigma_a*np.outer(y,np.transpose(s))) + sigma_a*(np.outer(s, np.transpose(s)))
            
        p= -np.dot(V,gradf(x0)) 
        
        pold = p
        a,h_ = bracketing(x0, a, phi(0,x0,pold,count), phi_(0,x0,pold),pold,count)
        # a,h_ = Backtrack(a,pold,x0,count)
        xold = x0
        x0 = x0 + a*p
        k=k+1
        # if (np.linalg.norm(phi_(0,x0,p)) < 0.001):# 0.82*np.linalg.norm(gradf(x0))): #0.85,0.7,0.8,0.9  for n=6 0.72 without line search, n=4 0.82, n=2 rosen 0.82 mu_2 =0.9 and rho = 0.8
        # if (phi_(0,x0,p) < 0.01):
        #     print("reset done!")
        #     reset = True
        # else :
        reset = False
        print("the Major iteration {} and the optimal value {} and \ngradf {}".format(k,x0,gradf(x0)))
        print("the optimal at the line search {}".format(a))
        print("infinity norm.{} \n  ".format(max(gradf(x0))))

        h_x.append(x0)
        # h_f.append(fun(x0,count))
        gradf_c.append(max(abs(gradf(x0))))
        iter_c.append(k+1)
    return x0, fun(x0,count),  np.array(h_x), np.array(h_f), np.array([1]),count, gradf_c, iter_c



### Conjugate gradient ####
def Nonlinear_CG(x0, tau=10**(-5), a_init=0.05):
    """
    Inputs- 
    x0 : starting point
    tau : Convergence tolerance

    Outputs - 
    xstar : Optimal point 
    fstar : Minimum function value
    """
    reset = True
    k =0
    a = a_init
    beta = 0
    # Keeping track of the history
    h_x=[x0]
    h_f = []
    while abs(max(gradf(x0))) > tau :
        if (k==0 or reset == True):
            
            print("Reset done!")
            # import pdb
            # pdb.set_trace()
            p = -gradf(x0)/(np.linalg.norm(gradf(x0)))

        else:
            a = a*abs((np.dot(np.transpose(gradf(x0)),pold))/(np.dot(np.transpose(gradf(x0+ a*pold)),p)))
            # beta = np.dot(np.transpose(gradf(x0)),gradf(x0))/np.dot(np.transpose(gradf(xold)),gradf(xold))
            # beta = (np.linalg.norm(gradf(x0))/np.linalg.norm(gradf(xold)))**2
            beta = np.dot(np.transpose(gradf(x0)),gradf(x0) - gradf(xold))/np.dot(np.transpose(gradf(xold)),gradf(xold))
            print("beta{}".format(beta))

            beta = np.maximum(0,beta)
            if (abs(np.dot(np.transpose(gradf(x0)),gradf(xold))/np.dot(np.transpose(gradf(x0)),gradf(x0))) >= 0.1):
                print("it's a reset{}".format(abs(np.dot(np.transpose(gradf(x0)),gradf(xold))/np.dot(np.transpose(gradf(x0)),gradf(x0)))))
                reset = True
                continue
            print("Just before {}".format(abs(np.dot(np.transpose(gradf(x0)),gradf(xold))/np.dot(np.transpose(gradf(x0)),gradf(x0)))))
            p = - gradf(x0)/(np.linalg.norm(gradf(x0))) + beta*pold
            print("new direction{}".format(p))
        pold = p
        # a,h_ = bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold)
        a,h_ = Backtrack(a,pold,x0)
        xold = x0
        x0 = x0 + a*p
        k = k+1
        reset= False
        # if (abs(np.dot(np.transpose(gradf(x0)),gradf(xold))/np.dot(np.transpose(gradf(x0)),gradf(x0))) >= 0.1):
        #     print("it's a reset")
        #     reset = True

        print("the Major iteration {} and the optimal value {} and \ngradf {} and factor {}".format(k,x0,gradf(x0), beta))
        print("the optimal at the line search {}".format(a))
        print("infinity norm.{}".format(max(gradf(x0))))

        h_x.append(x0)
        h_f.append(fun(x0))
    return x0, fun(x0),  np.array(h_x), np.array(h_f), np.array(h_)
        
## Bracktracking line search
def Backtrack(a_init, p, x0, count, mu_1=10**(-5), rho= 0.6):  #0.601, for n=6 Rosenbrock 0.603 mu_1= 10**(-4) n=8 -1,-1 0.3017, for n=6 BFGS 0.8 is good
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
    while (phi(a,x0,p,count) > phi(0,x0,p,count) + mu_1*a*phi_(0,x0,p)):
        #print("phi(a){} and phi(0) {}".format(phi(a,x0,p),phi(a,x0,p)))
        a = a* rho
        k=k+1
        
        print("line search iteration {} and a {}".format(k,a))
        #print("phi {} and phi0 {} and extra {}".format(phi(a,x0,p,count),phi(0,x0,p,count), mu_1*a*phi_(0,x0,p)))
        h_a.append(a)
        
        
    h_a = np.array(h_a)
    return a, h_a

## Bracketing phase for the line search 
def bracketing(x, a_init, Phi, Phi_,p,count, mu_1=10**(-5), mu_2 = 0.96, sigma= 2): #bracketing(x0, a, phi(0,x0,pold), phi_(0,x0,pold),pold)  ## n=6 mu2 = 0.39, sigma = 2.0201
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
        
        Phi_2 = phi(a_2,x ,p,count)
        h_a.append(a_2)
        if (Phi_2 >Phi_1 + mu_1*a_2*Phi_1_) or (~first and Phi_2 > Phi_1 ):
            Phi_2_ = phi_(a_2,x,p)
            
            a_star = pinpoint(x,a_1, a_2, Phi, Phi_, Phi_1, Phi_2, Phi_1_, Phi_2_,p,count)
            # return a_star, np.array(h_a.append(a_star))
            h_a.append(a_star)
            return a_star, h_a
        Phi_2_ = phi_(a_2,x,p)
        if (abs(Phi_2_) <= -mu_2*Phi_): #-mu_2*Phi_
            a_star = a_2
            h_a.append(a_star)
            return a_star, h_a
        elif Phi_2_ >= 0:
            a_star = pinpoint(x,a_2,a_1,Phi, Phi_, Phi_2, Phi_1,Phi_2_, Phi_1_,p,count)
            h_a.append(a_star)
            return a_star, h_a
            # return a_star, np.array(h_a.append(a_star))
        else:
            a_1 = a_2
            a_2 = sigma*a_2

        first = False

##### pinpointing
def pinpoint(x,a_low,a_high,Phi_0, Phi_0_, Phi_low, Phi_high, Phi_low_, Phi_high_,p, count,mu_1 = 10**(-5), mu_2=0.96): #NLCG diff n=2 0.419, 0.4199 0.4194
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
        # a_p =cubic(a_low, a_high, Phi_low,Phi_low_, Phi_high, Phi_high_)
        # print("a_p value{}".format(a_p))
        Phi_p = phi(a_p,x ,p,count)
        
        if Phi_p > Phi_0 + mu_1*a_p*Phi_0_ or Phi_p > Phi_low :
            # print("new high reached {}".format(a_p))
            a_high = a_p
            Phi_high = Phi_p # for the quadratic interpolation
            Phi_high_ = phi_(a_p,x,p)
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
    # b = np.array([Phi_low, Phi_high, Phi_low_])
    # A = np.array([[1, a_low, a_low**2],[1,a_high, a_high**2], [0, 1, 2*a_low]])
    # if (np.linalg.det(A)==0):
    #     print(A)
    # c = np.linalg.solve(A,b)
    # a_star = -c[1]/(2*c[2])
    a_star = (2*a_low *(Phi_high - Phi_low) + Phi_low_*(a_low**2 - a_high**2))/(2*(Phi_high-Phi_low + Phi_low_*(a_low - a_high)))
    return a_star

def SQP(x0,count,tau_opt= 10**(-4), tau_f=10**(-3)):
    """
    Inputs
    x0 : Starting point
    tau_opt : Optimality tolerance
    tau_f : feasibility

    Outputs
    x_star :Optimal point 
    f_star : Optimum Value
    """
    lam = 0 ; sigma = 0
    a_init =1 
    a = a_init
    reset = True
    # Evaluate f,g,h and gradf, Jh
    f = fun(x0,count)
    h = hun(x0)

    df = gradf(x0)
    Jh = Jach(x0)
    

    GradL = np.array([[df[0]],[df[1]]]) + np.dot(Jh.T,lam)
    # print(Jh.T + np.array([df]))
    # setting the array to save steps
    h_x = [x0]
    h_f = []
    
    k = 0
    while max(abs(GradL)) > tau_opt or max(abs(h))> tau_f :
        if k==0 or reset ==  True :
            HL = np.eye(np.shape(x0)[0])
        else  :
            s = x0 - xold
            # f = fun(x0)
            h = hun(x0)
            
            df = gradf(x0)
            Jh = Jach(x0)
            GradL = np.array([[df[0]],[df[1]]]) + np.dot(Jh.T,lam)
            y = GradL - Grad_old

            # damping 
            if np.dot(s,y) >= 0.2*(np.dot(s,np.dot(Hold,s))):
                theta = 1
            else :
                theta = (0.8*(np.dot(s,np.dot(Hold,s))))/(np.dot(s,np.dot(Hold,s)) - np.dot(s,y))
            r = theta*(np.array(y.ravel())) + (1-theta)*np.dot(Hold,s)
            
            HL  = Hold - (Hold@np.outer(s,s)@Hold)/(np.dot(s,np.dot(Hold,s))) + np.outer(r,r)/(np.dot(r,s))
            # print("Hessian {}".format(HL))
            # import pdb
            # pdb.set_trace()

        # QP subproblem
        p_s = QP(HL,Jh, GradL,h)

        p_lam = p_s[np.shape(HL)[0]:]
        p_x = p_s[:np.shape(HL)[0]]
        
        # update lambda

        lam = lam + p_lam
        GradL = np.array([[df[0]],[df[1]]]) + np.dot(Jh.T,lam) ## new 
        Grad_old = GradL
        pold = p_x
        pold = np.array(pold.ravel())
        # print("search direction {}".format(pold))
        a,h_ = bracketing(x0, a_init, phi(0,x0,pold,count), phi_(0,x0,pold),pold,count)
        # a,h_ = Backtrack(a,pold,x0,count)
        #saving HL values
        Hold = HL

        #update x
        xold = x0
        x0 = x0 + a*pold
        # print("the optimal value at this iteration{}".format(x0))
        reset = False
        
        ## Re-evaluate the function and Gradients
        # Grad_old = GradL
        df = gradf(x0)
        Jh = Jach(x0)
        # lam = np.array(lam.ravel())
        GradL = np.array([[df[0]],[df[1]]]) + np.dot(Jh.T,lam)
        h_x.append(x0)
        h_f.append(fun(x0,count))
        k = k+1
    return x0,fun(x0,count), np.array(h_x), np.array(h_f)





# QP subproblem
def QP(HL,Jh, GradL,h):
    """
    For solving Quadratic Programming
    Input -
    HL : Hessian of the Langrangian 
    Jh : the jacobian of the constraints
    GradL : Gradient of the Lagrangian
    h : function array ([nh x 1])
    """
    A = np.block([[HL, Jh.T],[Jh, np.zeros((np.shape(Jh)[0],np.shape(Jh)[0]))]])
    
    b = np.block([[-GradL], [-h]])
    # print(b)
    p_s = np.linalg.solve(A,b)
    return p_s

    
### Functions and Gradients computations ###
def hun(x):
    """
    constraint
    Output-
    h : [nh x 1]
    """
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    h1 = 0.25*(x[0]**2) + x[1]**2 -1
    I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - x[1]* (1/(c3/c4))
    
    # return  np.array([[g1],[g2]])
    return np.array([h1])

def Jach(x):
    """
    Jacobian of the constraints
    Output - 
    Jh = [nh x nx]

    """
    Jh1 = np.array([0.5*x[0], 2*x[1]])
    # Jg1 = np.array([])
    # Jg2 = np.array([])
    return np.block([[Jh1]])

def phi(a, x, p,count):
    """
    The function of alpha
    Input- 
    a : the alpha
    x : current position
    p : the direction of step

    Output-
    function of alpha
    """
    
    final = fun_(x + a*p,count)
    return final

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
   
    return np.dot(np.transpose(gradf_(x + a*p)),p)



def fun_(x,count):
    """
    The function to be optimized
    Input- 
    x : the input values

    Output-
    f : the function values
    """
    a = count[0]
    a = a+1
    count[0]= a
    #return x**2 + (x-1)**2 + 1
    # return 0.1*x[0]**6 - 1.5*x[0]**4 + 5*x[0]**2 + 0.1*x[1]**4 + 3*x[1]**2 - 9*x[1] + 0.5*x[0]*x[1]
    # return x[0]**2 + x[1]**2
    # return (1 -x[0])**2 + 100*(x[1] - x[0]**2)**2 #RosenBrock function
    # return (1 - x[0])**2 + (1 - x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2 #Bean function
    # f=0 # N=6 RosenBrock
    # for i in range(127):
    #     f = f + (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    # return f
    # beta = 1.5
    # return x[0]**2 + x[1]**2 - beta * x[0] * x[1]  # slanted function
    # mu = 10   exterior   
    # if max(0,0.25*x[0]**2 + x[1]**2 -1 ) == 0:
    #     return x[0] + 2*x[1]
    # else:
    #     return x[0] + 2*x[1] + 0.5*mu*(0.25*x[0]**2 + x[1]**2 -1)**2

    # mu = 1 #interior   
    # print(0.25*x[0]**2 + x[1]**2 -1)
    # if 0.25*x[0]**2 + x[1]**2 -1 < 0:
        
    #     return x[0] + 2*x[1] - mu*np.log(-0.25*x[0]**2 - x[1]**2 +1)
        
    # else:
    #     return x[0] + 2*x[1]
        


    mu = 4 
    return x[0] + 2*x[1]  + mu*(0.25*x[0]**2 + x[1]**2 -1) # bracketing
    # (2*(0.25*x[0]**2 + x[1]**2 -1))*(0.5*x[0])/(2*(np.sqrt((0.25*x[0]**2 + x[1]**2 -1)**2)))
    # (2*(0.25*x[0]**2 + x[1]**2 -1))*(2*x[1])/(2*(np.sqrt((0.25*x[0]**2 + x[1]**2 -1)**2)))
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 0.5
    I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - x[1]* (1/(c3/c4))
    final = 250*x[0] + 250*x[1] + (mu)*((g1)  + (g2)) 
    return final


def fun(x,count):
    """
    The function to be optimized
    Input- 
    x : the input values

    Output-
    f : the function values
    """
    a = count[0]
    a = a+1
    count[0]= a
    #return x**2 + (x-1)**2 + 1
    # return 0.1*x[0]**6 - 1.5*x[0]**4 + 5*x[0]**2 + 0.1*x[1]**4 + 3*x[1]**2 - 9*x[1] + 0.5*x[0]*x[1]
    # return x[0]**2 + x[1]**2
    # return (1 -x[0])**2 + 100*(x[1] - x[0]**2)**2 #RosenBrock function
    # return (1 - x[0])**2 + (1 - x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2 #Bean function
    # f=0 # N=6 RosenBrock
    # for i in range(127):
    #     f = f + (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    # return f
    # beta = 1.5
    # return x[0]**2 + x[1]**2 - beta * x[0] * x[1]  # slanted function
    # mu = 10   exterior   
    # if max(0,0.25*x[0]**2 + x[1]**2 -1 ) == 0:
    #     return x[0] + 2*x[1]
    # else:
    #     return x[0] + 2*x[1] + 0.5*mu*(0.25*x[0]**2 + x[1]**2 -1)**2

    # mu = 1 #interior   
    # print(0.25*x[0]**2 + x[1]**2 -1)
    # if 0.25*x[0]**2 + x[1]**2 -1 < 0:
        
    #     return x[0] + 2*x[1] - mu*np.log(-0.25*x[0]**2 - x[1]**2 +1)
        
    # else:
    #     return x[0] + 2*x[1]
        


    mu = 1
    return x[0] + 2*x[1] 
    # (2*(0.25*x[0]**2 + x[1]**2 -1))*(0.5*x[0])/(2*(np.sqrt((0.25*x[0]**2 + x[1]**2 -1)**2)))
    # (2*(0.25*x[0]**2 + x[1]**2 -1))*(2*x[1])/(2*(np.sqrt((0.25*x[0]**2 + x[1]**2 -1)**2)))
    # [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    # mu = 0.5
    # I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    # g1 = c1/c2 - I
    # g2 = c3/c4 - x[1]
    # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2) 
    # return final
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 0.5
    I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - x[1]* (1/(c3/c4))
    final = 250*x[0] + 250*x[1] #+ (mu)*((g1)  + (g2)) 
    return final


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
    # return (1 -X)**2 + 100*(Y - X**2)**2 #Rosenbrock function
    # return (1 - X)**2 + (1 - Y)**2 + 0.5*(2*Y - X**2)**2 #Bean function
    # f=0 # N=6 RosenBrock
    # for i in range(5):
    #     f = f + (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    # return f
    # beta = 1.5
    # return X**2 + Y**2 - beta * X * Y # slant
    # [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    # print(c4)
    # mu = 0.5
    # I = p*Y + q*X**3 + r*X
    # g1 = c1/c2 - I
    # g2 = c3/c4 - Y
    # final = 250*X + 250*Y + (mu/2)*((g1)**2  + (g2)**2) 
    # # final = 250*X + 250*Y + (mu*0.5)*((c1/(p*Y + q*X**3 + r*X) -c2)**2  + (c3/Y - c4)**2)
    # return final
    # mu = 10
    
    # return X + 2*Y + 0.5*mu*(0.25*X**2 + Y**2 -1)**2  # equality


    mu = 4
    return X+ 2*Y  + mu*(0.25*X**2 + Y**2 -1)
    # mu = 1
    # z = X
    # # Apply conditions to the grids
    # condition = 0.25*X**2 + Y**2 - 1 < 0
    # z = np.where(condition, X + 2*Y - mu*np.log(-0.25*X**2 - Y**2 + 1), X+ 2*Y)
    # z = np.where(condition, X+ 2*Y, X + 2*Y +0.5*mu*(0.25*X**2 + Y**2 -1)**2)
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 0.5
    I = p*Y + q*X**3 + r*X
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - Y* (1/(c3/c4))
    final = 250*X + 250*Y # + (mu)*((g1)  + (g2)) 
    
    return final

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
    # return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] -2, 200*(x[1] - x[0]**2)]) # Rosenvelt derivative

    # return np.array([-2 + 2*x[0] - 4*x[0]*x[1] + 2*x[0]**3, -2 + 6*x[1] - 2*x[0]**2 ]) # bean function
    # df = [] # N=6 RosenBrock
    # for i in range(128):
    #     if (i==127):
    #         df.append(200 * (x[i] - x[i-1]**2))
    #     else:
    #         df.append(-400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i]))
    # return np.array(df)
    # beta = 1.5
    # return np.array([2 * x[0] - beta * x[1],2 * x[1] - beta * x[0]]) # slant
    # mu = 10
    
    # return np.array([1+ mu*(0.5*x[0])*(0.25*x[0]**2 + x[1]**2 -1), 2 + mu*(2*x[1])*(0.25*x[0]**2 + x[1]**2 -1)])
    mu = 4
    
    return np.array([1 , 2 ])
    
    # #exterior
    # mu = 10
    # if max(0,0.25*x[0]**2 + x[1]**2 -1) == 0 :
    #     return np.array([1,2])
    # else :
    #     return np.array([1+ mu*(0.5*x[0])*(0.25*x[0]**2 + x[1]**2 -1), 2 + mu*(2*x[1])*(0.25*x[0]**2 + x[1]**2 -1)])
    
    # mu = 1  # interrio
    
    # if 0.25*x[0]**2 + x[1]**2 -1 < 0 :
    #     print(-0.25*x[0]**2 - x[1]**2 +1)
    #     final = np.array ([1 - mu*(-0.5*x[0])*(1/(-0.25*x[0]**2 - x[1]**2 +1)), 2 - mu*(-2*x[1])*(1/(-0.25*x[0]**2 - x[1]**2 +1))])
    #     # print(final)
    #     print('gradient value {}'.format(final))
    #     return final
    # else :
    #     return np.array([1,2])
    
    
    # # [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    # mu = 0.5
    # I = p*x[1] + q*((x[0])**3) + r*x[0]
    # # g1 = (1/c2)*(c1/I - c2)

    # # dg1 = (1/c2)*(-c1/(I**2)) 
    # # g2 = (1/c4)*(c3/x[1] -c4)

    # # dg2 = (1/c4)*(-c3/(x[1]**2))
    # g1 = c1/c2 - I
    # g2 = c3/c4 - x[1]
    
    
    # return np.array([250 + mu*0.5*(-2*(g1)*(3*q*x[0]**2 + r)), 250 + mu*0.5*(-2*(g1)*p - 2*(g2))])
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 0.5
    I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - x[1]* (1/(c3/c4))
    
    return np.array([250 , 250]) #- mu*(c2/c1)*(3*q*x[0]**2 + r)    - mu*((c4/c3) + p*(c2/c1))

def gradf_(x):
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
    # return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] -2, 200*(x[1] - x[0]**2)]) # Rosenvelt derivative

    # return np.array([-2 + 2*x[0] - 4*x[0]*x[1] + 2*x[0]**3, -2 + 6*x[1] - 2*x[0]**2 ]) # bean function
    # df = [] # N=6 RosenBrock
    # for i in range(128):
    #     if (i==127):
    #         df.append(200 * (x[i] - x[i-1]**2))
    #     else:
    #         df.append(-400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i]))
    # return np.array(df)
    # beta = 1.5
    # return np.array([2 * x[0] - beta * x[1],2 * x[1] - beta * x[0]]) # slant
    # mu = 10
    
    # return np.array([1+ mu*(0.5*x[0])*(0.25*x[0]**2 + x[1]**2 -1), 2 + mu*(2*x[1])*(0.25*x[0]**2 + x[1]**2 -1)])
    mu = 4     # for h^2 increasing mu bring inside the ellipse
    
    return np.array([1+ mu*(0.5*x[0]) , 2 + mu*(2*x[1]) ]) # works for bracketing for mu> 0.96
    # return np.array([1+ 2*mu*(0.5*x[0])*(0.25*x[0]**2 + x[1]**2 -1), 2 + 2*mu*(2*x[1])*(0.25*x[0]**2 + x[1]**2 -1)]) # works for backtracking
    
    # #exterior
    # mu = 10
    # if max(0,0.25*x[0]**2 + x[1]**2 -1) == 0 :
    #     return np.array([1,2])
    # else :
    #     return np.array([1+ mu*(0.5*x[0])*(0.25*x[0]**2 + x[1]**2 -1), 2 + mu*(2*x[1])*(0.25*x[0]**2 + x[1]**2 -1)])
    
    # mu = 1  # interrio
    
    # if 0.25*x[0]**2 + x[1]**2 -1 < 0 :
    #     print(-0.25*x[0]**2 - x[1]**2 +1)
    #     final = np.array ([1 - mu*(-0.5*x[0])*(1/(-0.25*x[0]**2 - x[1]**2 +1)), 2 - mu*(-2*x[1])*(1/(-0.25*x[0]**2 - x[1]**2 +1))])
    #     # print(final)
    #     print('gradient value {}'.format(final))
    #     return final
    # else :
    #     return np.array([1,2])
    
    
    # # [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    # mu = 0.5
    # I = p*x[1] + q*((x[0])**3) + r*x[0]
    # # g1 = (1/c2)*(c1/I - c2)

    # # dg1 = (1/c2)*(-c1/(I**2)) 
    # # g2 = (1/c4)*(c3/x[1] -c4)

    # # dg2 = (1/c4)*(-c3/(x[1]**2))
    # g1 = c1/c2 - I
    # g2 = c3/c4 - x[1]
    
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 0.5
    I = p*x[1] + q*x[0]**3 + r*x[0]
    # # g1 = (1/c2)*(c1/(I) -c2)
    # # g2 = (1/c4*(c3/x[1] - c4))
    # # final = 250*x[0] + 250*x[1] + (mu/2)*((g1)**2  + (g2)**2)
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - x[1]* (1/(c3/c4))
    
    return np.array([250 - mu*(c2/c1)*(3*q*x[0]**2 + r), 250 - mu*((c4/c3) + p*(c2/c1))])

if __name__=='__main__':
    xm = np.linspace(-1000,1000,200)
    ym = np.linspace(-1000,1000,200)
    Y,X = np.meshgrid(ym,xm)
    # fun1 = lambda x:(1 -x[0])**2 + 100*(x[1] - x[0]**2)**2
    # res = minimize(fun1, [0,0], method='BFGS', tol=1e-5,options={'gtol': 1e-6, 'disp': True})
    # print(res)
    plt.contour(X, Y, 250*X + 250*Y,60,cmap='Greys')
    [c1,c2,c3,c4,p,q,r] =[12500,200,6*(10**2), 116, (250**3)/12, (125)/6, ((250**2) * 125 *0.5)]
    mu = 10
    I = p*Y + q*X**3 + r*X
   
    g1 = 1 -  I *(1/(c1/c2))
    g2 = 1 - Y* (1/(c3/c4))
    plt.contour(X,Y, 1 -  I *(1/(c1/c2)), [0])
    plt.contour(X,Y,1 - Y* (1/(c3/c4)), [0])
    plt.plot(-1.7240946,   5.1724,'*')
    # plt.contour(X,Y,250*X + 250*Y + mu*(abs(1 -  I *(1/(c1/c2))) + abs(1 - Y* (1/(c3/c4)))), 100, cmap='hot')
    # plt.contour(X,Y, 0.25*X**2 + Y**2  -1, [0])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.contour(X,Y, X-1,[0])
    # plt.plot(np.array(h_x)[:,0],np.array(h_x)[:,1],'r-*')
    plt.show()