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
        x[j] = x[j] - dx

    return np.array(J)



def CenDiff(x,f,h=10**(-8)):
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
        
        x[j] = x[j] -2*dx

        vectorized_functions = np.vectorize(lambda func: func(x))
        ff = vectorized_functions(f)
        
        
        # f_ = func(x) # if using def based function defination
        dfdx = (f_ -ff)/(2*dx)
        J.append(dfdx)
        x[j] = x[j] + dx

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
    print(Complex(np.array([1.5]),fnew))
    # print(Complex(np.array([1.934]),f2))
    x = AD(1.5,1)
    y = AD(2,1)
    c = AD(3,0)
    # fad = AD.__sin__(x) + AD.__cos__(x)
    # fad = AD.__sqrt__(x)
    # fad = AD.__exp__(x)/(AD.__sqrt__(AD.__pow__(x,c)))
    # fad = AD.__pow__(AD.__sin__(x),c)
    fad =  AD.__exp__(x)/(AD.__sqrt__(AD.__pow__(AD.__sin__(x),c) + AD.__pow__(AD.__cos__(x),c)))
    # fad = A
    print(fad)
    # hrange = np.array([10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7),10**(-8),10**(-9), 10**(-10),10**(-11), 10**(-12),10**(-13), 10**(-14),10**(-15), 10**(-16),10**(-17), 10**(-18),10**(-19),10**(-20),10**(-21), 10**(-22)])
    # # hrange = np.arange(10**(-22),10**(-1), 10)
    
    # hrange2 = np.array([10**(-300),10**(-301), 10**(-302),10**(-303),10**(-304), 10**(-305), 10**(-306),10**(-307),10**(-308) ,10**(-309), 10**(-310),10**(-311), 10**(-312)])
   
    # ef = []
    # ecom = []
    # ecen = []
    # solf = []
    # solcom = []
    # solf1 = []
    # solcom = []
    # solc = []
    # for i in range(len(hrange
    #                    )):
    #     h = hrange[i]
    #     exact = 4.0534278938986201
    #     f= ForDiff(np.array([1.5]),fnew, h)
    #     solf.append(float(f))
    #     com = Complex(np.array([1.5]),fnew, h)
        
    #     solcom.append(float(com))

    #     c= CenDiff(np.array([1.5]),fnew, h)
    #     solc.append(float(c))

    #     ef.append(abs(float(f) - exact)/abs(exact))
    #     ecen.append(abs(float(c) - exact)/abs(exact))
    #     if abs(float(com) - exact)/(1+abs(exact)) <=10**(-16):
    #         ecom.append(10**(-16))
    #     else:
    #         ecom.append(abs(float(com) - exact)/(1+abs(exact)))

    # print("df_dx{}".format(solcom))
    # # x = np.flip(hrange)
    # x = hrange
    # plt.loglog(x,ef, label='forward difference',marker='o', linestyle='-', color='b')
    # plt.loglog(x,ecom, label='Complex step',marker='o', linestyle='-', color='r')
    # plt.loglog(x,ecen, label='Central difference',marker='o', linestyle='-', color='g')
    # plt.legend()
    # plt.ylim([10**(-18),100])
    # plt.ylabel(r"$Relative_Error_  \epsilon$")
    # plt.xlabel("step size (h)")
    # plt.show()


    # ef = []
    # ecom = []
    # ecen = []
    # solf = []
    # solcom = []
    # solf1 = []
    # solcom = []
    # solc = []
    # for i in range(len(hrange2
    #                    )):
    #     h = hrange2[i]
    #     exact = 4.0534278938986201
    #     f= ForDiff(np.array([1.5]),fnew, h)
    #     solf.append(float(f))
    #     com = Complex(np.array([1.5]),fnew, h)
        
    #     solcom.append(float(com))

    #     c= CenDiff(np.array([1.5]),fnew, h)
    #     solc.append(float(c))

    #     ef.append(abs(float(f) - exact)/abs(exact))
    #     ecen.append(abs(float(c) - exact)/abs(exact))
    #     if abs(float(com) - exact)/(1+abs(exact)) <=10**(-16):
    #         ecom.append(10**(-16))
    #     else:
    #         ecom.append(abs(float(com) - exact)/(1+abs(exact)))

    # print("df_dx{}".format(solcom))
    # # x = np.flip(hrange)
    # x = hrange2
    # plt.loglog(x,ef, label='forward difference',marker='o', linestyle='-', color='b')
    # plt.loglog(x,ecom, label='Complex step',marker='o', linestyle='-', color='r')
    # plt.loglog(x,ecen, label='Central difference',marker='o', linestyle='-', color='g')
    # plt.legend()
    # plt.ylim([10**(-18),10])
    # plt.ylabel(r"$Relative_Error_  \epsilon$")
    # plt.xlabel("step size (h)")
    # plt.show()



