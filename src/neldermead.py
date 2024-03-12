"""
Implementation of the Nelder-Mead Algorithm with a test case. 

By Joseph Ward March 9th 2024
"""

import numpy as np
from numpy import cos, sqrt
from scipy.optimize import minimize 
from visualizer import * 


def egg_carton(x):
    """Example function to demonstrate function"""
    return 0.1*x[0]**2+0.1*x[1]**2-cos(3*x[0])-cos(3*x[1])

def circle(x):
    return x[0]**2+x[1]**2-10

def neldermead(f, x0, taux=10**-4, tauf=10**-4, l=1):
    """Implementation of the Nelder-Mead algorithm"""
    
    # Generate Simplex
    n = len(x0)
    sj = [l/(n*sqrt(2))*(sqrt(n+1)-1)+l/sqrt(2), l/(n*sqrt(2))*(sqrt(n+1)-1)]
    x = [x0]
    for j in range(0,n):
        temp_x = x0.copy()
        for i in range(n):
            if j == i:
                temp_x[i] = x0[i] + sj[0]
            else:
                temp_x[i] = x0[i] + sj[1]
        x.append(temp_x)
    # Print points
    x = np.array(x)
    points = [x0]
    delf = []
    meanf = 0
    tempf = 0 
    deltax = 0
    deltaf = 0
    for i in range(n): meanf += f(x[i])
    for i in range(n): tempf = f(x[i]-meanf)
    deltaf = sqrt(tempf**2)/n+1
    delf.append(deltaf)
    # print(x)

    #While Loop
    deltax = 999
    deltaf = 999
    xprev = x0

    k = 0
    while (deltax > taux and deltaf > tauf):
        k += 1
        x = sorted(x, key=lambda xi: f(xi)) #Sort List
        
        xc = 1/n * sum(x[0:-1])
        xr = xc + (xc - x[-1])

        # Is reflected point the best? - Expansion
        if f(xr) < f(x[0]):
            # print("Expansion")
            xe = xc + 2*(xc-x[-1])
            # Is expanded point better?
            if f(xe) < f(x[0]):
                x[-1] = xe # Accept Expansion
            else:
                x[-1] = xr # Accept Reflection
        
        # Is reflected better than second worst? - Accept Reflection
        elif f(xr) <= f(x[-2]):
            # print("Reflection")
            x[-1] = xr
        
        # Is reflected worse than worst?
        else:
            if f(xr) > f(x[-1]):
                xic = xc - 0.5*(xc-x[-1]) #Inside Contraction
                # print("Inside")
                if f(xic) < f(x[-1]):
                    x[-1] = xic
                else:
                    for j in range(1,n):
                        x[j] = x[0] + 0.5 *(x[i]-x[0]) # Shrink
            else:
                xoc = xc + 0.5*(xc - x[-1]) #Outside Contraction
                # print("Outside")
                if f(xoc) < f(xr):
                    x[-1] = xoc
                else:
                    for j in range(1,n):
                        x[j] = x[0] + 0.5*(x[j]-x[0])

        # Has the function converged?
        # deltax = np.abs(np.linalg.norm(x[0])-np.linalg.norm(xprev))
        # deltaf = np.abs(f(x[0])-f(xprev)) 
        meanf = 0
        tempf = 0 
        deltax = 0
        deltaf = 0
        for i in range(n-1): deltax += (np.linalg.norm(x[i]-x[-1]))  
        for i in range(n): meanf += f(x[i])
        for i in range(n): tempf = f(x[i]-meanf)
        deltaf = sqrt(tempf**2)/n+1
        delf.append(deltaf)
        points.append(x[0])
        # print(x, "Convergence", deltax, deltaf)
    
    return x[0], k, points, delf      


def main():
    x0 = [0.9,0.9]
    f = egg_carton

    xstar, k, points, delf = neldermead(f,x0)
    print("NM", xstar, "k", k, "f", f(xstar))
    
    res1 = minimize(f, x0, method="Nelder-Mead", options={'xatol': 10**-4, 'fatol': 10**-4})
    print("Scipy", res1.x, "k", res1.nit, "f", res1.fun)

    res2 = minimize(f, x0)
    print("\nMinimize", res2.x, "k", res2.nit, "f", res2.fun)

    # xvalue = 2
    # yvalue = 2
    # n1 = 1000
    # n2 = 999

    # x1_vect = np.linspace(-xvalue,xvalue,n1)
    # x2_vect = np.linspace(-yvalue,yvalue,n2)
    # y_vect = np.zeros([n1,n2])

    # # Generate the height vector
    # for i in range(n1):
    #     for j in range(n2):
    #         x = [x1_vect[i],x2_vect[j]]
    #         y_vect[i,j] = f(x)

    # # Plot the curve
    # plt.figure("Graph Contour Plot")
    # # print("Xstar", xstar)
    # CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),100,linewidths=2) #Generate Contours
    # plt.plot(xstar[0],  xstar[1],   "ro")
    # # plt.text(xstar[0],  xstar[1],   "My NM", color="black", ha="center", va="bottom", size=16)
    # plt.plot(res1.x[0], res1.x[1],  "ro")
    # # plt.text(res1.x[0], res1.x[1],  "Scipy", color="black", ha="center", va="bottom", size=16)
    # plt.plot(x0[0], x0[1],  "yo")
    # # plt.text(x0[0], x0[1],  "Start", color="black", ha="center", va="bottom", size=16)
    # plt.xlabel("X1")
    # plt.ylabel("X2")

    # plt.colorbar()

    # desc_text="Convergence Plot"
    # xlabel="Iterations"
    # ylabel="Standard Deviation of the Function Value"

    # plt.figure(desc_text)
    
    # x_vect = np.arange(0,len(points),1)
    # y_vect = delf
    # # for x in range(len(x_vect)):
    # #     y_vect.append()
    
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    
    # # Check if should be log scale
    # vlength = round(len(y_vect)/6)
    # if (np.abs(y_vect[0]-y_vect[vlength])>100):
    #     plt.yscale('log')

    # plt.plot(x_vect,y_vect)
    # plt.show()

if __name__ == "__main__":
    main()
