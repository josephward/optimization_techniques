"""
Implementation of the Genetic Algorithm with a test case.

By Joseph Ward March 9th 2024
"""

import numpy as np
from random import randrange, shuffle
from visualizer import *

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

def distance(x1,x2):
    # Application of the distance forumla
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

def closeweighted(f, P):
    # Selects parents with the greatest values, then based on which parents are the closest to each other
    # Then has kids on a weighted line between them

    nextgen = []
    newP = []

    while len(P) > 1:
        # Select Parents
        mother = P[0]
        father = P[1]
        id = 1
        for i in range(2,len(P[2:])):
            if distance(mother, P[i]) < distance(mother, father):
                father = P[i]
                id = i
        P.pop(id)
        P.pop(0)

        motherval = f(mother)
        fatherval = f(father)
        # Generate perturbance
        xperturb = np.array([-(np.subtract(father,mother))[1], (np.subtract(father,mother))[0]])
        yperturb = np.array([-(np.subtract(mother,father))[1], (np.subtract(mother,father))[0]])
        # Generate kids
        child1 = mother + 0.75*abs(motherval/(motherval+fatherval))*(np.subtract(father,mother)) + 0.01*randrange(0,10)*xperturb
        child2 = father + 0.75*abs(fatherval/(motherval+fatherval))*(np.subtract(mother,father)) + 0.01*randrange(0,10)*yperturb

        # Update population
        newP.extend([mother, father, child1.tolist(), child2.tolist()])

        ## Show the birthing process
        # plt.figure()
        # plt.plot(child1[0],child1[1], 'bo')
        # plt.plot(child2[0],child2[1], 'bo')
        # plt.plot([father[0], mother[0]], [father[1], mother[1]], 'r-')
        # plt.plot(father[0],father[1], 'r*')
        # plt.plot(mother[0],mother[1], 'y*')
        # plt.show()

    return newP
    
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
    points = []
    for i in range(n):
        x = [round(randrange(-100,101)*0.1, 1), round(randrange(-100,101)*0.1, 1)]
        P.append(x)
    
    while k < 100:
        print(f"Generation {k}")
        points.append(P)
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
        P = closeweighted(f, P)

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

    return

if __name__ == "__main__":
    f = circle
    genetic_algorithm(f,20)

    # # Convert to numpy array
    # mother = np.array([2.,1.])
    # father = np.array([1.,1.])

    # # Generate Kids
    # motherval = f(mother)
    # fatherval = f(father)

    # child1 = mother + 0.75*abs(motherval/(motherval+fatherval))*(father-mother)
    # child2 = father + 0.75*abs(fatherval/(motherval+fatherval))*(mother-father)
    # print(child1, child2)
    # plt.plot(child1[0],child1[1], 'yo')
    # plt.plot(child2[0],child2[1], 'yo')

    # print("fm",father,mother)

    # xperturb = np.array([-(father-mother)[1], (father-mother)[0]])
    # yperturb = np.array([-(mother-father)[1], (mother-father)[0]])
    # print(xperturb,yperturb)
    
    # # perturb = [-np.subtract(father,mother)[1], np.subtract(father,mother)[0]]
    # # perturb = [x * 0.1*randrange(-10,10) for x in perturb]
    
    # print("fm",father,mother)

    # child1 = mother + 0.75*abs(motherval/(motherval+fatherval))*(father-mother) + 0.05*randrange(-10,10)*xperturb
    # child2 = father + 0.75*abs(fatherval/(motherval+fatherval))*(mother-father) - 0.05*randrange(-10,10)*yperturb

    # print(child1,child2,father,mother)

    
    # plt.ylim(0,3)
    # plt.xlim(0,3)
    # plt.plot(child1[0],child1[1], 'bo')
    # plt.plot(child2[0],child2[1], 'bo')
    # plt.plot(father[0],father[1], 'r*')
    # plt.plot(mother[0],mother[1], 'r*')
    # plt.show()