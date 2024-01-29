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
x1_vect = np.linspace(-5,5,n1)
x2_vect = np.linspace(-5,5,n2)

# Initial value and direction
init_loc = [-4,-4]
p = np.array([1,0])

# Example objective function
def f(x):
    return x[0]**2+x[1]**2

# Line Search inputs
phi0 = np.array([10])           #Initial Location (scalar or array)
init_alpha = np.array([1])      #Initial Step size
alpha_inc = 2                   #Alpha increase factor                     
u1 = 10**-4                     #Sufficient decrease factor
u2 = 0.5                        #Sufficient curvature factor


# Line Search functions
def bracketing():
    #phi(alpha) = f(xk+alpha in p)
    phi = phi0
    phi_prime = approx_fprime(phi0,f)
    alpha = init_alpha

    phi2 = phi + alpha
    phi2_prime = approx_fprime(phi0 + alpha,f)
    
    first = True
    while first:
        if (phi2 > phi + u1*alpha*phi_prime or (not first and phi2 > phi) ):
            print("Pinpoint1")
            alpha2 = alpha
            return alpha2
        
        if (np.absolute(phi2) <= -u2*phi_prime):
            print("Alpha prime")
            alpha2 = alpha
            return alpha2
        
        elif phi2_prime >= 0:
            print("Pinpoint2")
            alpha2 = alpha
            return alpha2

        else:
            alpha2 = alpha
            alpha2 = alpha_inc*alpha2

        first = False
    


""" To hide uninitialized functions
def pinpointing():
    pass

# For Homework 3
# After minimizing sufficently along a direction, change the direction
def change_direction():
    pass
"""

def graph_func():
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
    graph_func()
    # value = bracketing()
    # print("Alpha: ",value)
    pass

if __name__ == "__main__":
    main()