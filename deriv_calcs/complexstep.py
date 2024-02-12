"""
Solution for Derivatives using Complex Step

Created by Joseph Ward on Feb 12th, 2024
"""

import numpy as np

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


#Complex Step function
def complex_step(f,x,h=10**-3):
    return np.imag(f(x+complex(0,h)))/h

#Forward Finite Differencing
def forward_finite_diff(f,p,x,h=10**-3):
    """Calculates the derivative at x using forward finite difference"""
    return (f(x+h*p) - f(x))/h

def main():
    pass

if __name__ == "__main__":
    main()