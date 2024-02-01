import numpy as np
import matplotlib.pyplot as plt

### Defining the forward difference ###

## TODO: write a more generalized version for the vector of functions ## <- Done
 
def ForDiff(x,f,h=10**(-8)):
    """
    Inputs- 
    x: Point about which to compute the gradient
    f: Vector of functions of interest

    Outputs -
    J: jacobian of f with respect to x

    """
    x0 =x.copy()
    vectorized_functions = np.vectorize(lambda func: func(x0))
    f0 = vectorized_functions(f)
    # f0 = func(x0) # if using def based function defination
    nx = np.shape(x)[0]
    J = []
    
    for j in range(nx):
        dx = h*(1+ abs(x0[j]) ) #
        x[j] = x[j] + dx

        # function eval at perturbed point
        # Use np.vectorize to apply the vector of functions to the input value
        vectorized_functions = np.vectorize(lambda func: func(x))
        f_ = vectorized_functions(f)
        
        
        # f_ = func(x) # if using def based function defination
        dfdx = (f_ -f0)/dx
        J.append(dfdx)
        x = x - dx

    return np.array(J)



def func(x):
    """
    Input-
    x: input value

    Output- 
    f: function of interest
    """
    return np.exp(x)/(np.sqrt(np.sin(x)**3 + np.cos(x)**3))

### Complex step ###

def Complex(x,f,h =10**(-200)):
    """
    x: point about which to compute the gradient
    f: function of interest

    Output -
    J: Jacobian of f about x
    """
    x = x.astype(complex)
    x0 =x

    nx = np.shape(x)[0]
    J = []
    for i in range(nx):
        temp = x[i]
        x[i] = complex(temp,h)
        print(complex(2,3))
        vectorized_functions = np.vectorize(lambda func: func(x))
        f_ = vectorized_functions(f)
        f_ = np.array(f_)
        J.append((f_.imag)/h)
        x[i] = complex(x[i],-h)


    return np.array(J)
        
## Automatic Differentiation ###

# Defining class for operator overloading
class AD:
    def __init__(self,value, derivative):
        self.value =value
        self.derivative = derivative

    def __add__(self, other):
        if isinstance(other, AD):
            # check if 'other' is an instance of the same class
            value = self.value + other.value
            derivative = self.derivative + other.derivative
            return AD(value, derivative)
        else:
            raise TypeError("Unsupported operand type")

    def __mul__(self, other):
        if isinstance(other, AD):
            value = self.value * other.value 
            derivative = self.derivative*other.value + other.derivative*self.value
            return AD(value,derivative)
        else:
            raise TypeError("Unsupported operand type")

    def __sin__(self):    
        value = np.sin(self.value) 
        derivative = np.cos(self.value)*self.derivative
        return AD(value,derivative)
        

        
    def __cos__(self):
        value = np.cos(self.value) 
        derivative = -np.sin(self.value)*self.derivative
        return AD(value,derivative)
    
    def __exp__(self):
        value = np.exp(self.value)
        derivative = np.exp(self.value)
        return AD(value, derivative)
    
    def __sqrt__(self):
        value = np.sqrt(self.value)
        derivative = 1/(2*np.sqrt(self.value))*self.derivative
        return AD(value, derivative)
    
    def __truediv__(self,other):
        if other.value == 0:
            raise ZeroDivisionError("Division by zero not allowed")
        value = self.value/other.value
        derivative = (self.derivative*other.value - self.value*other.derivative)/((other.value)*(other.value))
        return AD(value, derivative)

    def __pow__(self,other):
        value = np.power(self.value,other.value)
        derivative = value*((other.value*self.derivative)/self.value + other.derivative*np.log(self.value))
        
        # derivative = other.value*np.power(self.value, other.value -1)
        return AD(value,derivative)  
    
    def __str__(self):
        return f"value= {self.value} and derivative = {self.derivative}"


## finite difference
def finite(x, u0= 1, h = 10**(-5)):
    """
    Input-
    x: input vector
    Output -
    df/dx : Jacobian
    """
    

    # solver for non-linear system
    dx = h*(1+ abs(x))
    # dx = h
    xh = x+dx
    u = Newton(x, u0)
    uh = Newton(x+dx, u0)
    

    df_dx = (func(x+dx,uh) - func(x,u))/dx

    return df_dx

def Newton(x,u0):
    """
    solving non-linear system
    """ 
    delta  = 1;  
    while abs(delta)>10**(-3):
        unew = u0 - resi(x,u0)/dresi(x,u0)
        delta = unew - u0
        print("delta {}".format(delta))
        u0 = unew
    print("value of E {}".format(unew))
    return unew

def resi(x,u,e=1):
    return u -e*np.sin(u) - x 

def dresi(x,u,e=1):

    return 1 - e*np.cos(u) 

def func(x,u):
    return  u - x



if __name__=="__main__":

    # Create a list of functions using lambda function
    f = [
        lambda x: 2 * x[0] + 2*x[1],
        lambda x: x[0]**2 + x[1]**2,
        lambda x: np.sin(x[0]) + np.sin(x[1]),
        lambda x: np.exp(x[0])/(np.sqrt(np.sin(x[0])**3 + np.cos(x[0])**3)) + np.exp(x[1])/(np.sqrt(np.sin(x[1])**3 + np.cos(x[1])**3))
    ]

    fnew = [
        
        lambda x: np.exp(x[0])/(np.sqrt(np.sin(x[0])**3 + np.cos(x[0])**3))
    ]

    f2 = [
        lambda x: 1/(1 -np.cos(x))
    ]
    
    # print(ForDiff(np.array([1.5]),fnew, 10**(-8)))
    print(Complex(np.array([1.5,1.2]),f))
    # print(Complex(np.array([1.934]),f2))
    print(finite(1))
    x = AD(1.5,1)
    y = AD(2,1)
    c = AD(3,0)
    # fad = AD.__sin__(x) + AD.__cos__(x)
    # fad = AD.__sqrt__(x)
    # fad = AD.__exp__(x)/(AD.__sqrt__(AD.__pow__(x,c)))
    # fad = AD.__pow__(AD.__sin__(x),c)
    fad =  AD.__exp__(x)/(AD.__sqrt__(AD.__pow__(AD.__sin__(x),c) + AD.__pow__(AD.__cos__(x),c)))
    # fad = A
    # print(fad)
    print(- 1 - 1/(np.cos(1.934) -1))

    A = np.array([[1, 0,0],[1, np.cos(1.934) -1, 0],[1, -1,1]])
    b = np.array([[1,0,0]])
    print("initial A{}".format(A))
    print(np.shape(A))
    print(np.shape(b))

    
    rm =  [
        
        lambda x : 1.934 - np.sin(1.934) - x[0]
    ]
    re = [
        lambda x : x - np.sin(x) - 1
    ]

    fm =  [
        
        lambda x : 1.934 - x[0]
    ]
    fe = [
        lambda x: x[0] - 1
    ]

    # A = np.array([[1, 0,0],[float(-Complex(np.array([1]),rm)), float(-Complex(np.array([1.934]),re)), 0],[float(-Complex(np.array([1]),fm)), float(-Complex(np.array([1.934]),fe)),1]])
    print("A {}".format(A))
    print( -ForDiff(np.array([1.934]),re))
    print("Analytical {}".format( np.cos(1.934) -1))
    x = np.linalg.solve(A,b.T)
    print("solutions {}".format(x))
    print("adjoint {}".format(float(ForDiff(np.array([1]),fm)) -  float(ForDiff(np.array([1.934]),fe))/float(ForDiff(np.array([1.934]),re)) *float(ForDiff(np.array([1]),rm)) ))




