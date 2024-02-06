"""
Created on Jan 27th for Homework 2 and 3 for ME 575
by Joseph Ward

Line Search Algorithm
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time #Used for testing

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

def linesearch(f, f_prime, init_loc, search_type, tau=10**-5,
               u1=10**-4, u2=10**-2, sigma=1.5, init_alpha=1):
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
        while True:
            # Convergence Condition
            spot_prime = f_prime(xk)
            if (np.linalg.norm(spot_prime) < tau):
                break

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
        first = True
        f_grad = float()
        prior_f_grad = float()
        p = []
        prior_p = []
        xk = init_loc
        alpha = init_alpha
        Bk = 0

        while True:
            np.append(search_points, xk)
            
            #Start with steepest descent
            if (first == True or reset == True):
                first = False
                reset = False
                f_grad = f_prime(xk)
                #Check tau condition
                if (np.linalg.norm(f_grad) < tau):
                    break
                p = f_grad/-np.linalg.norm(f_grad)
                prior_p = p

            #Continue with Conjugate Gradient
            else:
                prior_f_grad = f_grad
                f_grad = f_prime(xk)
                Bk = (f_grad*np.transpose(f_grad))/(prior_f_grad*np.transpose(prior_f_grad))
                #Check tau condition
                if (np.linalg.norm(f_grad) < tau):
                    break
                prior_p = p
                p = f_grad/-np.linalg.norm(f_grad) + Bk*prior_p

            alpha = bracketing(f, f_prime, xk, p, u1, u2, sigma, alpha)
            xk = xk + alpha*p
            k += 1
            
            #Check if time to reset
            print(f_grad, prior_f_grad)
            top = np.dot(f_grad,prior_f_grad)
            bottom = np.dot(f_grad,f_grad)
            reset_var = np.abs(top/bottom)
            print(reset_var)
            if (reset_var >= 0.1):
                reset = True

    elif (search_type=="QN"):
        pass
    else:
        errortext = "Must select one of the following search direction algorithims: 'SD' (Steepest Descent), 'CG' (Conjugate Gradient), or 'QN' (Quasi-Newton)"
        raise ValueError(errortext) #SD, CG, or QN
    
    # TODO: Do I want k to be a global variable or not?
    
    print("\n\nFunction: ", f)
    print("Min Location: ",xf)
    print("Min Value: ",res)
    print("Function Calls: ", k)
    print("Searches: ", len(search_points))

    return res, xf, k, search_points

def bracketing(f, f_prime, x0, p, u1=10**-4, u2=10**-1, sigma=1.5, init_alpha=1):
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
                                  u1=10**-4, u2=10**-1, sigma=1.5, init_alpha=1)
            return alphastar
        
        if (np.abs(phi2_prime) <= -u2*phi_prime):
            # print("Alpha prime")
            alphastar = alpha2
            return alphastar
        
        elif (phi2_prime >= 0):
            # print("Pinpoint2")
            alphastar = pinpoint(f, f_prime, x0, p, alpha2, alpha1, 
                              u1=10**-4, u2=10**-1, sigma=1.5, init_alpha=1)
            return alphastar

        else:
            alpha1 = alpha2
            alpha2 = sigma*alpha2
        first = False

def pinpoint(f, f_prime, x0, p, alpha_low, alpha_high, u1=10**-4, u2=10**-1, sigma=1.5, init_alpha=1):
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

def interpolate(f,f_prime,x0,p,alpha1,alpha2):
    top = (2*alpha1*(f(x0+alpha2*p)-f(x0+alpha1*p)))+calc_phiprime(f_prime(x0+alpha1*p,p)*(alpha1**2-alpha2**2))
    bottom = 2*(f(x0+alpha2*p)-f(x0+alpha1*p)+calc_phiprime(f_prime(x0+alpha1*p,p)*(alpha1**2-alpha2**2)))
    alphastar = top/bottom
    return alphastar

def only_graph(f,init_loc,n1=100,n2=99):
    """Generates just the graph of the function, inital value, and global minimum.
    
    Parameters:
        f (function):       Objective Function.
        init_loc (list):    Initial location, which is graphed.
    Returns:
        Graph of objective function, initial location, and global minimum.
    """

    x1_vect = np.linspace(-10,10,n1)
    x2_vect = np.linspace(-10,10,n2)
    y_vect = np.zeros([n1,n2])

    # Generate the height vector
    for i in range(n1):
        for j in range(n2):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)
            
    res = minimize(f,init_loc)

    # Plot the curve
    plt.figure("Graph Contour Plot")
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),100,linewidths=2) #Generate Contours
    #Initial Point
    plt.plot(init_loc[0],init_loc[1],"bo")
    plt.annotate("Initial Point",[init_loc[0],init_loc[1]])
    #Global Min
    plt.plot(res.x[0],res.x[1],"ro")
    plt.annotate("Global Min Point",[res.x[0],res.x[1]])
    plt.colorbar()

def graph_linesearch(f,init_loc,search_points,n1=100,n2=99):
    """
    Graphs the result of the linesearch.

    Parameters:
        f (function):           Objective function.
        init_loc (list):        List of coordinates of initial condition.
        search_points (list):   List of points arrived at during the linesearch
        
        n1 (int):               X resolution of graph
        n2 (int):               Y resolution of graph

    Returns:
        Graph of linesearch with each substep

    """
    
    # Graph down to the dimensions of the linesearch 
    maxx = 0
    minx = 999
    maxy = 0
    miny = 999
    for i in range(len(search_points)):
        tempx = search_points[i][0]
        tempy = search_points[i][1]
        if (tempx > maxx):
            maxx = round(tempx,2)
        if (tempx < minx):
            minx = round(tempx,2)
        if (tempy > maxy):
            maxy = round(tempy,2)
        if (tempy < miny):
            miny = round(tempy,2)

    # Generate the graphing space
    spacing_var = 0.1*(maxx-minx)
    x1_vect = np.linspace(round(minx,2)-spacing_var,round(maxx,2)+spacing_var,n1)
    x2_vect = np.linspace(round(miny,2)-spacing_var,round(maxy,2)+spacing_var,n2)
    y_vect = np.zeros([n1,n2])

    # Generate the height vector
    for i in range(n1):
        for j in range(n2):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)

    # Plot the curve
    plt.figure("Graph Only Function")
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),25,linewidths=2) #Generate Contours
    plt.clabel(CS, inline=True, fontsize=10)

    # Plot global min point
    res = minimize(f,init_loc) # Find minimum
    print("Actual Min: ",res.x)
    plt.plot(res.x[0],res.x[1],"r*") #Plot minimum
    plt.annotate("Scipy Min",[res.x[0],res.x[1]])

    # Plot linesearch values
    plt.plot()
    for i in range(len(search_points)-1):
        plt.plot([search_points[i][0],search_points[i+1][0]],[search_points[i][1],search_points[i+1][1]],"ro-")

    #Annotate Graph
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()

# Graph phi as a function of alpha
def graph_slice():
    pass

def main():

    #Homework 2 Functions and Initial Values
    #              0            1           2           3
    func_list   = [SQ,          RB,         J,          bean]
    dir_list    = [SQ_prime,    RB_prime,   J_prime,    bean_prime]
    loc_list    = [[2,-6],      [0,2],      [1,1],      [2,3]]
    i = 1
    res, x, k, points = linesearch(func_list[i],dir_list[i],loc_list[i],SEARCH_DIRECTION_ALG[1])

    graph_linesearch(func_list[i],loc_list[i],points)
    # only_graph(func_list[i],loc_list[i])
    plt.show()

if __name__ == "__main__":
    main()