"""
Solves a multi-objective optimization and graphs the pareto front.
Uses the Binh and Korn function as an example.
"""
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
from random import randrange

random.seed(None)

def f1(x):
    """Min f(0,0)"""
    return 4*x[0]**2 + 4*x[1]**2

def f2(x):
    """Min f(0,5)"""
    return (x[0] - 5)**2 + (x[1] - 5)**2

def RB(x):
    """Rosenbrock Function - D.1.2"""
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2


def g1(x):
    return -((x[0] - 5)**2 + x[1]**2 - 25)

def g2(x):
    return ((x[0] - 8)**2 + (x[1] + 3)**2 - 7.7)

def pareto_opt(f1, f2, n=100):
    front = []
    points = []

    # # Develop the pareto front
    # N = 10
    # for w in range(N):
    #     w = w/N
    #     res = minimize(lambda x: w*f1(x)+(1-w)*f2(x), [-10,-10])
    #     print(res.x)
    #     front.append(res.x)

    

    # # Develop the rest of the points
    # N = 50

    # # Develop the pareto front
    # N = 10
    # for w in range(N):
    #     w = w/N
    #     res = minimize(lambda x: w*f1(x)+(1-w)*f2(x), [-10,-10], constraints=constraints)
    #     print(res.x)
    #     points.append(res.x)

    # # Develop Body out 
    # N = 30
    # for i in range(N):
    #     x = [randrange(-10,10)/10, randrange(-10,10)/10]
    #     res =[f1(x), f2(x)]
    #     print(res)
    #     points.append(res)


    # Develop pareto front
    N = 10
    x0 = [-10, -10]
    # General Constraints
    constraints = []
    # constraints = [{'type': 'ineq', 'fun': g1}, {'type': 'ineq', 'fun': g2}]

    # Solve for f1
    constraints.append({'type': 'ineq', 'fun': lambda x: x[0] - e})
    
    for e in range(0,161,10):
        res = minimize(f1, x0, constraints=constraints)
        front.append(res.x)
    constraints.pop()
    
    # # Solve for f2
    # constraints.append({'type': 'ineq', 'fun': lambda x: x[1] - e})
    # for e in range(N):
    #     res = minimize(f2, x0, constraints=constraints)
    #     front.append(res.x)

    # Develop Body out 
    N = 50
    for i in range(N):
        x = [randrange(-50,50)/10, randrange(-50,50)/10]
        res =[f1(x), f2(x)]
        print(res)
        points.append(res)

    return points, front

def main():
    points, front = pareto_opt(f1,f2)
    
    plt.figure("Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")

    plt.plot([point[0] for point in front], [point[1] for point in front],"ro")
    plt.plot([point[0] for point in points], [point[1] for point in points],"bo")

    plt.show()

if __name__ == "__main__":
    main()