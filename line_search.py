"""
Created on Jan 27th for Homework 2 and 3 for ME 575
by Joseph Ward

Line Search Algorithm
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

# Graph variables
n1 = 100
n2 = 99
x1_vect = np.linspace(-10,10,n1)
x2_vect = np.linspace(-10,10,n2)

# Initial value and direction
init_loc = np.array([-4,-4])
p = np.array([1,1])
# Line Search inputs
phi0 = 0                        #Initial Location
phi0_prime = 0                  #Initial Gradient
init_alpha = 1                  #Initial Step size
sigma = 1.9                     #Alpha increase factor                     
u1 = 10**-6                     #Sufficient decrease factor
u2 = 10**-2                     #Sufficient curvature factor
k = 0

# Example objective function
def h(x):
    return x[0]**2+x[1]**2

def h_prime(x):
    return 2*x[0]+2*x[1]

# Slanted Quadratic
def SQ(x):
    return x[0]**2+x[1]**2-1.5*x[0]*x[1]

def SQ_prime(x):
    return 2*x[0]+2*x[1]-1.5*(x[0]+x[1])

# Rosenbrock
def RB(x):
    return 0

def RB_prime(x):
    return 0

# Jones
def J(x):
    return 0

def J_prime(x):
    return 0

# Given a direction, it finds the optimal point along the line
def linesearch1(f, f_prime, x0, p):
    alpha = bracketing(f, f_prime, x0, p)

    xf = x0 + alpha*p
    val = f(xf)

    return xf, val

def bracketing(f, f_prime, x0, p):
    #Calculate initial values
    phi = f(x0)
    phi_prime = f_prime(x0)
    phi1 = phi
    phi1_prime = phi_prime
    alpha1 = 0
    alpha2 = init_alpha
    
    first = True
    while True:
        #Take a guess
        phi2 = f(x0+alpha2*p)
        phi2_prime = f_prime(x0+alpha2*p)

        print(x0,alpha2,x0+alpha2*p)

        #Does the guess satisify the strong wolfe conditions?
        #If phi is above the line 
        if (phi2 > phi + u1*alpha2*phi_prime or (not first and phi2 > phi1) ):
            print("Pinpoint1")
            alpha_star = pinpoint(f, f_prime, x0, p, alpha1, alpha2)
            return alpha_star
        
        if (np.absolute(phi2_prime) <= -u2*phi_prime):
            print("Alpha prime")
            alpha2 = alpha2
            return alpha2
        
        elif phi2_prime >= 0:
            print("Pinpoint2")
            alpha2 = pinpoint(f, f_prime, x0, p, alpha2, alpha1)
            return alpha2

        else:
            alpha1 = alpha2
            alpha2 = sigma*alpha2
        first = False
    
def pinpoint(f, f_prime, x0, p, alpha_low, alpha_high):
    while True:
        global k
        k += 1 #Iterate Count

        # Recalc values
        alpha_p = (alpha_low + alpha_high)/2
        phi = f(x0)
        phi_prime = f_prime(x0)
        phip = f(x0+alpha_p*p)
        phip_prime = f_prime(x0+alpha_p*p)
        philow = f(x0+alpha_low*p)
        philow_prime = f_prime(x0+alpha_low*p)
        phihigh = f(x0+alpha_high*p)
        phihigh_prime = f_prime(x0+alpha_high*p)
        
        # alpha_p is above the line
        if (phip > phi + u1*alpha_p*phi0_prime or phip > philow):
            print("Move upper lower")
            # if(k<10):
            #     print(phip, phi+u1*alpha_p*phi0_prime)
            alpha_high = alpha_p

        else:
            # It is close enough based on u2
            if (np.absolute(phip_prime) <= -u2*phi_prime):
                print("Just right")
                alphastar = alpha_p
                return alphastar
            
            # Need to look further
            elif (phip_prime*(alpha_high-alpha_low)>=0):
                print("Not between")
                alpha_high = alpha_low

            # Move up the low value
            alpha_low = alpha_p
            print("Move lower higher")


""" To hide uninitialized functions

# For Homework 3
# After minimizing sufficently along a direction, change the direction
def change_direction():
    pass
"""

# TODO: Make a plot of equation 4.27
def graph_func(f):
    global n1,n2
    y_vect = np.zeros([n1,n2])

    # Generate the height vector
    for i in range(n1):
        for j in range(n2):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)
            
    
    # print(res)

    # Plot the curve
    plt.figure("Graph Contour Plot")
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),10) #Generate Contours
    plt.clabel(CS, inline=True, fontsize=10)
    
    #Annotate Graph
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    # plt.colorbar()

    # Plot minimum point
    res = minimize(f,init_loc) # Find minimum
    plt.plot(res.x[0],res.x[1],"r*") #Plot minimum
    plt.annotate("Min",[res.x[0],res.x[1]])

    # Plot initial point
    plt.arrow(init_loc[0],init_loc[1],p[0],p[1],head_width=0.25)
    # plt.plot([init_loc[0],init_loc[0]+p[0]],[init_loc[1],init_loc[1]+p[1]],"ro-") #Unit vector in p
    plt.plot(init_loc[0],init_loc[1],"bo",) #Initial Point
    plt.annotate("Initial Point",[init_loc[0],init_loc[1]])


    ### TODO: Develop a graph of the slice along p
    # # Plot the subsection in the p direction
    # plt.figure("Subsection")

    # # From -5 to 5, graph the curve along p
    # sub_x = np.linspace(-5,5,100)
    # sub_y = []

    # print(len(sub_x),len(sub_y),len(p))
    # for i in range(len(sub_x)):
    #     print(sub_x[i]*np.transpose(p[0]),sub_x[i]*np.transpose(p[1]))
    #     x = [sub_x[i]*np.transpose(p[0]),sub_x[i]*np.transpose(p[1])]
    #     sub_y.append(f(x))

    # plt.plot(sub_x,sub_y)
    # plt.plot(init_loc[0],f(init_loc),"bo")

    # plt.xlabel("x")
    # plt.ylabel("Height")
    # plt.grid()


    plt.show()

def main():       
    global init_loc, p, phi0, phi0_prime, init_alpha, sigma, u1, u2, k
    
    # h, SQ, RB, J and the ..._prime functions
    func = h
    func_dir = h_prime

    init_loc = np.array([-4,-4])
    p = np.array([1,1])
    # Line Search inputs
    phi0 = func(init_loc)               #Initial Location
    phi0_prime = func_dir(init_loc)     #Initial Gradient
    init_alpha = 1                      #Initial Step size
    sigma = 1.9                         #Alpha increase factor                     
    u1 = 10**-6                         #Sufficient decrease factor
    u2 = 10**-4                         #Sufficient curvature factor


    x, res = linesearch1(func,func_dir,init_loc,p)
    print("Function: ", func)
    print("Min Location: ",x)
    print("Min Value: ",res)
    print("K: ", k)
    graph_func(func)

if __name__ == "__main__":
    main()