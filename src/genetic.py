"""
Implementation of the Genetic Algorithm with a test case.

By Joseph Ward March 9th 2024
"""

import numpy as np
from random import randrange, shuffle

# Global Variables
k = 0 

def egg_carton(x):
    """Example function to demonstrate function"""
    return 0.1*x[0]**2+0.1*x[1]**2-np.cos(3*x[0])-np.cos(3*x[1])

def circle(x):
    return x[0]**2+x[1]**2

def cull(f,P):
    """ 
    Perfectly balanced, as all things should be
    Remember principles of: Elitism, Tournament, Roulette, Random Parents
    This takes the top 20%, and random 1 or 2 values, and conducts a tournement with the rest
    """
    pop = len(P)

    # Elitism
    # Save the top 20%
    n = len(P)
    topP = P[0:round(n*0.2)]
    P = P[round(n*0.2):]
    # shuffle(P)

    # Roulette
    # Also ensures that P is an even number of points
    n = len(P)
    randomP = []
    if(n % 2 != 0):
        randomP.append(P[0])
        P = P[1:]
    else:
        randomP.append(P[0])
        randomP.append(P[1])
        P = P[2:]
    
    # Tournament
    # Compares half of the population against the other half of the population, and removes the worst ones
    n = len(P)
    tournyP = []
    for i in range(int(n/2)):
        tournyP.append(max(P[0:2], key=lambda x: f(x)))
        P = P[2:]

    print(topP, randomP, tournyP)
    newP = topP + randomP + tournyP
    P = newP[0:round(pop/2)] #Make sure that 50% were culled
    return P


def closeweighted(P):
    # Selects parents with the greatest values, then based on which parents are the closest to each other
    # Then has kids on a weighted line between them
    pass
    
def greatest(f, P):
    # Selects the parents with the greatest value, then has kids on a line between them.
    pass

def mutate(P):
    # Slightly mutates the population
    pass

def genetic_algorithm(f,n):
    """Implementation of the Genetic Algorithm algorithm"""
    global k
    ## Generate Population
    P = []
    for i in range(n):
        x = [round(randrange(-100,101)*0.1, 1), round(randrange(-100,101)*0.1, 1)]
        P.append(x)
    
    while k < 100:
        print(f"Generation {k}")
        #Evaluate obj function
        tempf = []
        P.sort(key=lambda x: f(x))
        for i in range(len(P)):
            tempf.append(f(P[i]))

        print("Population", P)
        print("Func Evals", tempf)

        # Cull population
        P = cull(f, P)

        # Select parents and generate new generation
        P = greatest(f, P)

        # TESTING
        print("\nPopulation", P)
        tempf = []
        print(P[0][0])
        for i in range(len(P)):
            tempf.append(f(P[i]))
        print("Func Evals", tempf)
        # TESTING

        while True: return

        # Mutate
        P = mutate(P)

        k += 1
        print("\n")
        return 

    pass

if __name__ == "__main__":
    f = circle
    genetic_algorithm(f,10)