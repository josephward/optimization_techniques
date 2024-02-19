"""
Created on Jan 27th for Homework 2 and 3 for ME 575
by Joseph Ward

Line Search Algorithm
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time #Used for testing
from visualizer import *

# Global Variables
k = 0

#### Objective Functions

# Circle Function
def h(x):
    """Simple Circle Function"""
    global k
    k += 1 #Iterate Count
    return x[0]**2+x[1]**2
def h_prime(x):
    """Derivative of Simple Circle Function"""
    prime = [2*x[0],2*x[1]]
    return prime

# Slanted Quadratic
def SQ(x):
    """Slanted Quadratic Function - D.1.1"""
    global k
    k += 1 #Iterate Count
    return x[0]**2+x[1]**2-1.5*x[0]*x[1]
def SQ_prime(x):
    """Derivative of Slanted Quadratic Function - D.1.1"""
    prime = np.array([2*x[0]-1.5*x[1],2*x[1]-1.5*x[0]])
    return prime

# Rosenbrock
def RB(x):
    """Rosenbrock Function - D.1.2"""
    global k
    k += 1 #Iterate Count
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2
def RB_prime(x):
    """Derivative of Rosenbrock Function - D.1.2"""
    prime = np.array([(-2*(1-x[0]))-400*x[0]*(x[1]-x[0]**2),200*(x[1]-x[0]**2)])
    return prime

# Bean Function
def bean(x):
    """Bean Function - D.1.3"""
    global k
    k += 1 #Iterate Count
    return (1-x[0])**2+(1-x[1])**2+0.5*(2*x[1]-x[0]**2)**2
def bean_prime(x):
    """Derivative of Bean Function - D.1.3"""
    prime_x = -2*(1-x[0])-2*x[0]*(2*x[1]-x[0]**2)
    prime_y = -2*(1-x[1])+2*(2*x[1]-x[0]**2)
    prime = np.array([prime_x,prime_y])
    return prime

# Jones Function
def J(x):
    """Jones Function - D.1.4"""
    global k
    k += 1 #Iterate Count
    return x[0]**4 + x[1]**4 - 4*x[0]**3 - 3*x[1]**3 + 2*x[0]**2 +2*x[0]*x[1]
def J_prime(x):
    """Derivative of Jones Function - D.1.4"""
    prime_x = 4*(x[0]**3) -12*(x[0]**2) +4*x[0] +2*x[1]
    prime_y = 4*(x[1]**3) -9*x[1]**2 +2*x[0]
    prime = np.array([prime_x,prime_y])
    return prime

#### Calculation and Graphing Functions

# Calculate Phi Prime for Prime Functions
def calc_phiprime(prime,p):
    """Calculates phi value for derivative functions"""
    return np.dot(prime,p)

SEARCH_DIRECTION_ALG = ["SD", "CG", "QN"] #Steepest Descent, Conjugate Gradient, Quasi-Newton

# def linesearch(f, f_prime, init_loc, search_type, tau=10**-3,
#                u1=10**-4, u2=.25, sigma=1.8, init_alpha=0.5):
def linesearch(f, f_prime, init_loc, search_type, tau=10**-3,
               u1=10**-4, u2=0.5, sigma=1.5, init_alpha=1):
    """
    Conducts a line search optimization for the function f, starting at location init_loc, in direction of p.
    
    Linesearch overload which requires an analytical solution in the form of a function.

    Parameters:
        f (function):                   Objective function.
        f_prime (function):             Analytical solution to objective function.
        init_loc (list):                N dimensional list of numbers for starting location.
        search_type(string):            Search direction algorithim selector.
        tau (float):                    Tolerance parameter. The magnitude of the p vector must be smaller than tau 

        u1 (float):                     First Strong Wolfe condition, specifies the line of sufficient decrease.
        u2 (float):                     Second Strong Wolfe condition, specifies tolerance for the sufficient curvature condition.
        sigma (float):                  Specifies change in alpha each bracketing loop.
        init_alpha (float):             Specifies initial alpha value.

    Returns:
        res (float):                    Value of optimization.
        x (list):                       N dimensional list of ints of the location.
        k (int):                        Number of objective function calls.
        search_points (numpy array):    A list of the initial, intermediate, and final points from the search.

    """        
    
    global k

    # Build return variables
    xf = []
    search_points = []
    k = 0

    # Run search direction algorithm
    if (search_type=="SD"):
        alpha = init_alpha #testing, replace with estimate alpha
        xk = init_loc
        search_points.append(xk)
        while np.linalg.norm(f_prime(xk),np.inf) > tau:
            # Convergence Condition
            spot_prime = f_prime(xk)
            # print("\nNew Linesearch: Loc", init_loc, "Norm", np.linalg.norm(spot_prime))
            np.append(search_points, xk)
            p = spot_prime/-np.linalg.norm(spot_prime)
            #estimate alpha
            alpha = bracketing(f, f_prime, xk, p, u1, u2, sigma, alpha)
            xk = xk + alpha*p
            search_points.append(xk)

        xf = xk
        res = f(xf)
        
    elif (search_type=="CG"):
        #Set up variables
        xk = init_loc
        alpha = init_alpha
        f_grad = float()
        prior_f_grad = float()
        Bk = float()
        p = []
        prior_p = []
        reset = False
        reset_points = []

        while (np.linalg.norm(f_prime(xk),np.inf) > tau):
            search_points.append(xk)
            prior_f_grad = f_grad
            f_grad = f_prime(xk)
            prior_p = p

            #Start with steepest descent
            if (len(search_points) == 1 or reset == True):
                # print("Start First or Reset")
                reset = False
                p = f_grad/-np.linalg.norm(f_grad)
                # print("First or reset:",xk)
                reset_points.append(xk)

            #Continue with Conjugate Gradient
            else:
                # print("Start Else")
                Bk = np.dot(f_grad,f_grad)/np.dot(prior_f_grad,prior_f_grad)
                p = f_grad/-np.linalg.norm(f_grad) + Bk*prior_p
                # print("Else:",xk)
            
            # Check if you need to reset
            reset_var = np.abs(np.dot(f_grad,prior_f_grad)/np.dot(f_grad,f_grad))
            if (len(search_points) > 1 and reset_var >= 0.1):
                reset = True

            alpha = bracketing(f, f_prime, xk, p, u1, u2, sigma, alpha)
            # print("End B")
            xk = xk + alpha*p
        
        xf = xk
        res = f(xf)
        print(search_type,"Reset",len(reset_points))

    elif (search_type=="QN"):
        
        # Set up variables
        xk = init_loc
        alpha = init_alpha
        n = len(init_loc) # Number of Dimensions
        reset = False

        Vk = np.identity(n)
        prior_Vk = np.identity(n)
        f_grad = float()
        prior_f_grad = float()
        search_points = []
        reset_points = []

        while (np.linalg.norm(f_prime(xk),np.inf) > tau):
            search_points.append(xk)
            prior_f_grad = f_grad
            f_grad = f_prime(xk)
            prior_Vk = Vk

            if (len(search_points) == 1 or reset == True):
                reset = False
                Vk = np.divide(1,np.linalg.norm(f_grad))*np.identity(n)
                # print("Init",Vk)
                reset_points.append(xk)

            else:
                s = np.subtract(xk,search_points[-2])
                y = np.subtract(f_grad,prior_f_grad)
                o = 1/(np.dot(s,y))
                Vk = np.subtract(np.identity(n),o*s*np.transpose(y)) * prior_Vk * np.subtract(np.identity(n),o*y*np.transpose(s)) + o*s*np.transpose(s)
            
            p = np.dot(-Vk,f_grad)

            # Check if you need to reset
            reset_var = np.abs(np.dot(f_grad,prior_f_grad)/np.dot(f_grad,f_grad))
            if (len(search_points) > 1 and reset_var >= 0.1):
                reset = True

            # if (len(search_points) > 1 and len(search_points)%6 == 0):
            #     reset = True

            alpha = bracketing(f, f_prime, xk, p, u1, u2, sigma, alpha)
            xk = xk + alpha*p

        xf = xk
        res = f(xf)
        print(search_type, "reset",len(reset_points))

    else:
        errortext = "Must select one of the following search direction algorithims: 'SD' (Steepest Descent), 'CG' (Conjugate Gradient), or 'QN' (Quasi-Newton)"
        raise ValueError(errortext) #SD, CG, or QN
    
    # TODO: Do I want k to be a global variable or not?
    
    print("\n\nFunction:", f)
    print("Method:", search_type)
    print("Min Location:",xf)
    print("Min Value:",res)
    print("Function Calls:", k)
    print("Searches:", len(search_points))

    return res, xf, k, search_points

def testing(f, f_prime, init_loc, search_type, tau=10**-2,
               u1=10**-4, u2=.25, sigma=1.5, init_alpha=1):
    # Set up variables
        search_points = [init_loc]
        reset = False
        f_grad = float()
        prior_f_grad = float()
        Vk = np.identity(1)
        prior_Vk = np.identity(1)
        alpha = init_alpha
        xk = init_loc
        n = len(init_loc)
        print("n",n)
        

        search_points.append(xk)
        prior_f_grad = f_grad
        f_grad = f_prime(xk)

        # First step
        print(np.linalg.norm(f_grad))
        print(np.divide(1,np.linalg.norm(f_grad)))
        print(np.identity(n))
        Vk = np.divide(1,np.linalg.norm(f_grad))*np.identity(n)
        print("Vk",Vk)

        p = np.dot(-Vk,f_grad)
        print("p",p)
        alpha = bracketing(f, f_prime, xk, p, u1, u2, sigma, alpha)
        xk = xk + alpha*p

        # Second step
        s = np.subtract(xk,search_points[-1])
        print("s",s)
        y = f_grad - prior_f_grad
        print("y",y)
        o = 1/(np.dot(s,y))
        print("o",o)
        # Vk = ()
        print("Vk",Vk)
        # print("osy", o*s*np.transpose(y))
        # print("I osy", np.subtract(np.transpose(n),o*s*np.transpose(y)))
        # print("oys", o*y*np.transpose(s))
        # print("oss", o*s*np.transpose(s))
        # print("All", np.subtract(np.transpose(n),o*s*np.transpose(y)) * prior_Vk * np.subtract(np.transpose(n),o*y*np.transpose(s)) + o*s*np.transpose(s))
        Vk = np.subtract(np.transpose(n),o*s*np.transpose(y)) * prior_Vk * np.subtract(np.transpose(n),o*y*np.transpose(s)) + o*s*np.transpose(s)
        print("Vk",Vk)
        
        # V = (np.identity() - np.transpose(o*s*y))*prior_Vk*()

def bracketing(f, f_prime, x0, p, u1, u2, sigma, init_alpha):
    """
    Bracketing algorithm (4.3) which establishes a min and max bound beneath the line of sufficient decrease.
    Following that, that value is passed to a pinpointing algorithm, the alpha of the location of the minimum is calculated, and returned.

    Parameters:
        f (function):           Objective Function.
        f_prime (function):     Analytical derivative of objective function.
        x0 (list):              Initial location.
        p (list):               Search direction from search direction.

        u1 (float):             First Strong Wolfe condition, specifies the line of sufficient decrease.
        u2 (float):             Second Strong Wolfe condition, specifies tolerance for the sufficient curvature condition.
        sigma (float):          Specifies change in alpha each bracketing loop.
        init_alpha (float):     Specifies initial alpha value.

    Returns:
        alphastar (float):      The distance along p from x0 where the minimum is found.
    """

    #Calculate initial values
    phi = f(x0)
    phi_prime = calc_phiprime(f_prime(x0),p)
    phi1 = phi
    phi1_prime = phi_prime
    alpha1 = 0
    alpha2 = init_alpha
    
    first = True
    while True:
        #Take a guess
        phi2 = f(x0+alpha2*p)
        phi2_prime = calc_phiprime(f_prime(x0+alpha2*p),p)

        #Does the guess satisify the strong wolfe conditions?
        #If phi is above the line 
        val = u1*alpha2*phi_prime
        if (phi2 > phi + val or (first == False and phi2 > phi1) ):
            # print("Pinpoint1")
            alphastar = pinpoint(f, f_prime, x0, p, alpha1, alpha2, 
                                  u1, u2, sigma, init_alpha)
            return alphastar
        
        if (np.abs(phi2_prime) <= -u2*phi_prime):
            # print("Alpha prime")
            alphastar = alpha2
            return alphastar
        
        elif (phi2_prime >= 0):
            # print("Pinpoint2")
            alphastar = pinpoint(f, f_prime, x0, p, alpha2, alpha1, 
                              u1, u2, sigma, init_alpha)
            return alphastar

        else:
            alpha1 = alpha2
            alpha2 = sigma*alpha2

        # if (abs(alpha1-alpha2) < 0.00001):
        #     return alpha1
        first = False

def pinpoint(f, f_prime, x0, p, alpha_low, alpha_high, u1, u2, sigma, init_alpha):
    """
    Pinpointing algorithm (4.4) to find the local minimum between a bracket. The alpha of that location is returned to the bracketing function.

    Parameters:
        f (function):           Objective Function.
        f_prime (function):     Analytical derivative of objective function.
        x0 (list):              Initial location.
        p (list):               Search direction from search direction.
        alpha_low (float):      Lower alpha value.
        alpha_high (float):     High alpha value.

        u1 (float):             First Strong Wolfe condition, specifies the line of sufficient decrease.
        u2 (float):             Second Strong Wolfe condition, specifies tolerance for the sufficient curvature condition.
        sigma (float):          Specifies change in alpha each bracketing loop.
        init_alpha (float):     Specifies initial alpha value.

    Returns:
        alphastar (float):      The distance along p from x0 where the minimum is found.
    """

    phi = f(x0)
    phi_prime = calc_phiprime(f_prime(x0),p)
    
    while True:        
        # Recalc values
        # Bisection method or interpolation
        alpha_p = (alpha_low + alpha_high)/2
        # alpha_p = interpolate(f,f_prime,x0,p,alpha_low,alpha_high)
        phip = f(x0+alpha_p*p)
        phip_prime = calc_phiprime(f_prime(x0+alpha_p*p),p)

        phi_low = f(x0+alpha_low*p)
        phi_high = f(x0+alpha_high*p)
        
        # alpha_p is above the line
        if (phip > phi + u1*alpha_p*phi_prime or phip > phi_low):
            # print("Move upper lower", phi_low, phip, phi_high)
            alpha_high = alpha_p

        else:
            # It is close enough based on u2
            if (np.absolute(phip_prime) <= -u2*phi_prime):
                # print("Just right",phi_low, phip, phi_high)
                alphastar = alpha_p
                return alphastar
            
            # Need to look further
            elif (phip_prime*(alpha_high-alpha_low)>=0):
                # print("Not between",phi_low, phip, phi_high)
                alpha_high = alpha_low

            # Move up the low value
            alpha_low = alpha_p
            # print("Move lower higher",phi_low, phip, phi_high)
            # time.sleep(0.25)

        # if (abs(alpha_high-alpha_low) < 0.00001):
        #     return alpha_p

def interpolate(f,f_prime,x0,p,alpha1,alpha2):
    top = (2*alpha1*(f(x0+alpha2*p)-f(x0+alpha1*p)))+calc_phiprime(f_prime(x0+alpha1*p,p)*(alpha1**2-alpha2**2))
    bottom = 2*(f(x0+alpha2*p)-f(x0+alpha1*p)+calc_phiprime(f_prime(x0+alpha1*p,p)*(alpha1**2-alpha2**2)))
    alphastar = top/bottom
    return alphastar

def main():

    #Homework 2 Functions and Initial Values
    #              0            1           2           3
    func_list   = [SQ,          RB,         J,          bean]
    dir_list    = [SQ_prime,    RB_prime,   J_prime,    bean_prime]
    loc_list    = [[2,-6],      [0,2],      [1,1],      [2,3]]

    # testing(func_list[i],dir_list[i],loc_list[i],SEARCH_DIRECTION_ALG[0])
    
    res_list = []
    i = 1
    j = 1
    res, x, k, points = linesearch(func_list[i],dir_list[i],loc_list[i],SEARCH_DIRECTION_ALG[j],tau=10**-4,u1=10**-3,u2=0.5)
    graph_linesearch(func_list[i],loc_list[i],points)
    # plot_convergence(dir_list[i],points)
    plt.show()

if __name__ == "__main__":
    main()