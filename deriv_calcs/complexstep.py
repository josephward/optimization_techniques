"""
Solution for Derivatives using Complex Step

Created by Joseph Ward on Feb 12th, 2024
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt 
k = 0

#Functions
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
# Rosenbrock Function
def RB(x):
    """Rosenbrock Function - D.1.2"""
    global k
    k += 1 #Iterate Count
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2
def RB_prime(x):
    """Derivative of Rosenbrock Function - D.1.2"""
    prime = np.array([(-2*(1-x[0]))-400*x[0]*(x[1]-x[0]**2),200*(x[1]-x[0]**2)])
    return prime

# Four Dim. Rosenbrock Function
def RB4d(x):
    one     = 100*(x[1]-x[0]**2)**2 + (1+x[0])**2
    two     = 100*(x[2]-x[1]**2)**2 + (1+x[1])**2
    three   = 100*(x[3]-x[2]**2)**2 + (1+x[2])**2
    return one+two+three

def RB4d_prime(x):
    one     = 400*x[0]**3 + 2*x[0] - 400*x[1]*x[0] + 2
    two     = 400*x[1]**3 + 202*x[1] - 400*x[1]*x[2] + 2 - 200*x[0]**2
    three   = 400*x[2]**3 + 202*x[2] - 400*x[2]*x[3] + 2 - 200*x[1]**2
    four    = 200*x[3] - 200*x[2]**2
    prime   = np.array([one, two, three, four])
    return prime

#Complex Step function
def complex_step(f,x,h=10**-200):
    """Calculates the jacobian using complex step"""
    J = np.zeros(len(x))
    x = x.astype(complex)
    for j in range(len(x)):
        x[j] = x[j] + complex(0,h)
        fp = f(x)
        J[j] = np.imag(fp)/h
        x[j] = x[j] - complex(0,h)
    return J

#Forward Finite Differencing
def forward_finite_diff(f,x,h=10**-10):
    """Calculates the jacobian using forward finite difference"""
    f0 = f(x)
    J = np.zeros(len(x))
    for j in range(len(x)):
        dx = h*(1+np.abs(x[j]))
        x[j] = x[j] + dx
        fp = f(x)
        J[j] = (fp - f0)/dx
        x[j] = x[j] - dx
    return J


def main():
    x0 = np.array([0.5,-0.5,0.5,1])
    h_lim = 10**-30
    h = 10**-1
    cj_diff = []
    fj_diff = []
    x_vect = []
    while (h > h_lim):
        analyticJ = RB4d_prime(x0)
        complexJ = complex_step(RB4d,x0,h)
        forwardJ = forward_finite_diff(RB4d,x0,h)
        cj_diff.append(abs(np.linalg.norm(analyticJ)-np.linalg.norm(complexJ)))
        fj_diff.append(abs(np.linalg.norm(analyticJ)-np.linalg.norm(forwardJ)))
        x_vect.append(str("{0:.1e}".format(h)))

        # print(h)
        # print("Analytical",analyticJ)
        # print("Complex Step",complexJ)
        # print("Forward Finite Diff",forwardJ)
        # print("Norm Complex vs Forward", np.linalg.norm(complexJ)-np.linalg.norm(forwardJ))
        # print("")

        h = h*10**-1

    plt.figure("Complex vs Forward")
    plt.plot(x_vect,cj_diff,"-bo")
    print(x_vect)
    print(cj_diff)
    plt.plot(x_vect,fj_diff,"-ro")
    plt.xticks(rotation=90)    
    plt.xlabel("Value of h")
    plt.ylabel("Difference of Norms")
    plt.yscale("log")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()