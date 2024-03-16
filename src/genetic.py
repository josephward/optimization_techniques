"""
Implementation of the Genetic Algorithm with a test case.

By Joseph Ward March 9th 2024
"""

import numpy as np
from random import *
from visualizer import *
from time import sleep

# Global Variables
k = 0 
seed(10) # Set the seed for testing

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

    while len(P) > 0:
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

        # # Show the birthing process
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

def mutate(P,y=0.01):
    # Slightly mutates the population

    for i in range(len(P)):
        if randrange(0,100) <= round(100*y):
            P[i] = [P[i][0]+randrange(-100,101)*0.01,P[i][1]+randrange(-100,101)*0.01]
            # print("Mutated",i)

    return P

def visualprint(points):
    for j in range(len(points[0])):
            print(' '.join(map(str, points[:,j,:].tolist())))

def genetic_algorithm(f,iter=100,n=20):
    """Implementation of the Genetic Algorithm algorithm"""
    assert n % 2 == 0, "n must be an even number"
    assert (n // 2) % 2 == 0, "n divided by 2 must be an even number"
    
    global k
    ## Generate Population and prepopulate the points
    P = []
    points = np.zeros((n, iter, 2))
    if n % 4 == 0:
        for i in range(n//4):
            x = [round(randrange(0,101)*0.1, 1), round(randrange(0,101)*0.1, 1)]
            P.append(x)
            x = [round(randrange(0,101)*0.1, 1), round(randrange(-100,1)*0.1, 1)]
            P.append(x)
            x = [round(randrange(-100,1)*0.1, 1), round(randrange(0,101)*0.1, 1)]
            P.append(x)
            x = [round(randrange(-100,1)*0.1, 1), round(randrange(-100,1)*0.1, 1)]
            P.append(x)
    else:
        for i in range(n):
            x = [round(randrange(-100,101)*0.1, 1), round(randrange(-100,101)*0.1, 1)]
            P.append(x)

    while k < iter:
        #Evaluate obj function
        tempf = []
        for i in range(n):
            points[i,k] = P[i]
        P.sort(key=lambda x: f(x))
        for i in range(len(P)):
            tempf.append(f(P[i]))

        # Cull population
        P = cull(f, P)

        # Mutate
        P = mutate(P)

        # Select parents and generate new generation
        P = closeweighted(f, P)

        # # TESTING
        # print("\nPopulation", P)
        # tempf = []
        # for i in range(len(P)):
        #     tempf.append(f(P[i]))
        # print("Func Evals", tempf)
        # # TESTING

        
        # print("\n")
        k += 1
    P.sort(key=lambda x: f(x))
    return P, points

def main():
    # f = circle
    f = egg_carton
    iter = 100
    n = 100
    P, points = genetic_algorithm(f,iter, n) 

    # ### Testing
    # # Calculate the average point
    # avg_point = np.mean([point[-1] for point in points], axis=0)
    # print("Average Point:", avg_point, f.__name__, f(avg_point))

    print("Minimum:", [0,0], f.__name__, f([0,0]))

    # return
    # ### Testing

    # Graph the function
    plt.figure("Beginning Points")

    # Plot the beginning points
    for i in range(len(P)):
        plt.plot(points[i][0][0], points[i][0][1], "b*") 

    # Plot the countour
    x1_vect = np.linspace(-10,10,100)
    x2_vect = np.linspace(-10,10,99)
    y_vect = np.zeros([100,99])

    # Generate the height vector
    for i in range(100):
        for j in range(99):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)

    plt.contour(x1_vect,x2_vect,np.transpose(y_vect))
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Graph the function
    plt.figure("Last Point")

    # Plot the end points
    for i in range(len(P)):
        plt.plot(points[i][-1][0], points[i][-1][1], "r*")

    # Calculate the average point
    avg_point = np.mean([point[-1] for point in points], axis=0)

    # Plot the average point
    plt.plot(avg_point[0], avg_point[1], "go")
    print("Average Point:", avg_point, f.__name__, f(avg_point))

    # Plot the countour
    x1_vect = np.linspace(-10,10,100)
    x2_vect = np.linspace(-10,10,99)
    y_vect = np.zeros([100,99])

    # Generate the height vector
    for i in range(100):
        for j in range(99):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)

    plt.contour(x1_vect,x2_vect,np.transpose(y_vect))
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()

    """Plot in a loop"""

    # for j in range(iter):
    #     # Plot the beginning points
    #     plt.figure(f.__name__ + " Iteration " + str(j))

    #     for i in range(len(P)):
    #         if j != 0:
    #             plt.plot([points[i][j-1][0], points[i][j][0]], [points[i][j-1][1], points[i][j][1]], 'ro')
    #         plt.plot(points[i][j][0], points[i][j][1], "b*") 

    #     # Plot the countour
    #     x1_vect = np.linspace(-10,10,100)
    #     x2_vect = np.linspace(-10,10,99)
    #     y_vect = np.zeros([100,99])

    #     # Generate the height vector
    #     for i in range(100):
    #         for j in range(99):
    #             x = [x1_vect[i],x2_vect[j]]
    #             y_vect[i,j] = f(x)

    #     plt.contour(x1_vect,x2_vect,np.transpose(y_vect))
    #     plt.colorbar()
    #     plt.xlabel("x1")
    #     plt.ylabel("x2")

    #     plt.show()
    return

if __name__ == "__main__":
    main()