"""
Created on Jan 27th for Homework 2 and 3 for ME 575
by Joseph Ward

Line Search Algorithm
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
import time

# Graph variables
n1 = 100
n2 = 99
x1_vect = np.linspace(-5,5,n1)
x2_vect = np.linspace(-5,5,n2)

# Initial value and direction
init_loc = np.array([0,0])
p = np.array([1,1])
# Line Search inputs
phi0 = 0                        #Initial Location
phi0_prime = 0                  #Initial Gradient
init_alpha = 1                  #Initial Step size
sigma = 0                       #Alpha increase factor                     
u1 = 0                          #Sufficient decrease factor
u2 = 0                          #Sufficient curvature factor
k = 0

# Example objective function
def h(x):
    global k
    k += 1 #Iterate Count
    return x[0]**2+x[1]**2

def h_prime(x,p):
    prime = [2*x[0],2*x[1]]
    prime = np.dot(prime,p)
    return prime

# Slanted Quadratic
def SQ(x):
    global k
    k += 1 #Iterate Count
    return x[0]**2+x[1]**2-1.5*x[0]*x[1]

def SQ_prime(x,p):
    prime = np.array([2*x[0]-1.5*x[1],2*x[1]-1.5*x[0]])
    prime = np.dot(prime,p)
    return prime

# Rosenbrock
def RB(x):
    global k
    k += 1 #Iterate Count
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def RB_prime(x,p):
    prime = np.array([(-2*(1-x[0]))-400*x[0]*(x[1]-x[0]**2),200*(x[1]-x[0]**2)])
    prime = np.dot(prime,p)
    return prime

# Jones
def J(x):
    global k
    k += 1 #Iterate Count
    return (1-x[0])**2+(1-x[1])**2+0.5*(2*x[1]-x[0]**2)**2

def J_prime(x,p):
    prime_x = -2*(1-x[0])-2*x[0]*(2*x[1]-x[0]**2)
    prime_y = -2*(1-x[1])+2*(2*x[1]-x[0]**2)
    prime = np.array([prime_x,prime_y])
    prime = np.dot(prime,p)
    return prime

# Given a direction, it finds the optimal point along the line
def linesearch1(f, f_prime, x0, p):
    alpha = bracketing(f, f_prime, x0, p)

    xf = x0 + alpha*p
    val = f(xf)

    return xf, val, alpha

def bracketing(f, f_prime, x0, p):
    
    #Calculate initial values
    phi = f(x0)
    phi_prime = f_prime(x0,p)
    phi1 = phi
    phi1_prime = phi_prime
    alpha1 = 0
    alpha2 = init_alpha
    
    first = True
    while True:
        #Take a guess
        phi2 = f(x0+alpha2*p)
        phi2_prime = f_prime(x0+alpha2*p,p)

        #Does the guess satisify the strong wolfe conditions?
        #If phi is above the line 
        val = u1*alpha2*phi_prime
        if (phi2 > phi + val or (first == False and phi2 > phi1) ):
            print("Pinpoint1")
            alpha_star = pinpoint(f, f_prime, x0, p, alpha1, alpha2)
            return alpha_star
        
        print(phi2_prime,-u2*phi_prime)
        if (np.abs(phi2_prime) <= -u2*phi_prime):
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
    
    phi = f(x0)
    phi_prime = f_prime(x0,p)
    
    while True:
        # Recalc values
        alpha_p = (alpha_low + alpha_high)/2
        phip = f(x0+alpha_p*p)
        phip_prime = f_prime(x0+alpha_p*p,p)

        phi_low = f(x0+alpha_low*p)
        
        # alpha_p is above the line
        if (phip > phi + u1*alpha_p*phi_prime or phip > phi_low):
            print("Move upper lower")
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
def graph_func(f,x_sol,res,alphastar):
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
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),100,linewidths=2) #Generate Contours
    # plt.clabel(CS, inline=True, fontsize=10)
    
    #Annotate Graph
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.colorbar()

    # Plot minimum point
    res = minimize(f,init_loc) # Find minimum
    plt.plot(res.x[0],res.x[1],"r*") #Plot minimum
    plt.annotate("Min",[res.x[0],res.x[1]])

    # Plot initial point and vector away from it
    # plt.arrow(init_loc[0],init_loc[1],p[0],p[1],head_width=0.25)
    plt.plot(init_loc[0],init_loc[1],"bo",) #Initial Point
    plt.annotate("Initial Point",[init_loc[0],init_loc[1]])

    # Plot results from solver
    print(x_sol)
    plt.plot(x_sol[0],x_sol[1],"b*")
    plt.annotate("Min Along Line",[x_sol[0],x_sol[1]])
    s_ans = x_sol+alphastar*p
    plt.plot([init_loc[0],s_ans[0]],[init_loc[1],s_ans[1]],"b-")


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
    
    # h, SQ, RB, J and their ..._prime functions
    func = J
    func_dir = J_prime
    p = np.array([1,-3])
    init_loc = np.array([0,2])

    # Line Search inputs
    phi0 = func(init_loc)               #Initial Location
    phi0_prime = func_dir(init_loc,p)   #Initial Gradient
    init_alpha = 1                      #Initial Step size
    sigma = 1.5                           #Alpha increase factor                     
    u1 = 10**-4                         #Sufficient decrease factor
    u2 = .1                            #Sufficient curvature factor


    x, res, alphastar = linesearch1(func,func_dir,init_loc,p)
    print("\n\nFunction: ", func)
    print("Min Location: ",x)
    print("Min Value: ",res)
    print("Alpha Star: ", alphastar)
    print("K: ", k)
    graph_func(func,x,res,alphastar)

if __name__ == "__main__":
    main()