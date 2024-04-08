import numpy as np
import matplotlib.pyplot as plt
import random

# Traveling Salesman Problem Generation

def generate_points():
    """Generate random points, and ensure they are unique."""
    flag = True
    while flag:
        points = np.vstack([[0, 0], np.random.randint(-50, 50, size=(49, 2))])
        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] == points.shape[0]:
            flag = False
        else:
            print("Duplicate points generated, retrying...")
    return points

def greedy_alg(points):
    """Greedy algorithm implementation."""
    start = np.array(points[0]) # Starting point, (0,0)
    to_visit = points[1:] # Remove starting point
    path = np.array([start]) # Used for graphing later
    travel_distance = 0
    current_point = start
    convergence = []

    while to_visit[:,0].size > 0:
        closest_point = to_visit[0]
        dist_loc = 0
        
        for i in range(to_visit[:,0].size):
            temp_dist = distance(current_point, to_visit[i])
            if temp_dist < distance(current_point, closest_point):
                closest_point = to_visit[i]
                dist_loc = i

        to_visit = np.delete(to_visit, dist_loc, axis=0)
        current_point = closest_point
        path = np.append(path, np.array([current_point]), axis=0)
        travel_distance = path_distance(path)
        convergence.append(travel_distance)
    
    path = np.append(path, np.array([start]), axis=0) # Return to starting point
    travel_distance += distance(current_point, start)
    convergence.append(travel_distance)

    return path, travel_distance, convergence

def plot(points,path,name=None):
    plt.figure(name)
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot(path[:, 0], path[:, 1], 'r-')
    plt.plot(0, 0, 'ro')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-55, 55)
    plt.ylim(-55, 55)

def distance(point1, point2):
    """Calculate the distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def path_distance(path):
    dis = 0
    for i in range(path[:,0].size-1):
        dis += distance(path[i], path[i+1])
        # print(i, i+1)
    return dis

def gen_neightbor(path):
    """Generate a neighbor path for simulated annealing.
    This will not swap the first and last points, because the path needs to return to the same point."""    
    int1 = 0
    int2 = 0
    while int1 == int2:
        int1 = random.randint(1, path[:,0].size-2)
        int2 = random.randint(1, path[:,0].size-2)
    point1 = path[int1]
    point2 = path[int2]
    new_path = np.copy(path)
    new_path[int1] = point2
    new_path[int2] = point1

    return new_path

def sim_annealing(path, kmax, T, alpha, beta=150):
    """Simulated annealing function"""

    convergence_dist = [path_distance(path)]

    for k in range(kmax):
        temp_path = gen_neightbor(path)

        if path_distance(temp_path) <= path_distance(path):
            path = temp_path

        else: 
            r = random.random()
            # print(path_distance(temp_path), path_distance(path), T, r)
            P = np.exp(-(path_distance(temp_path) - path_distance(path))/T)
            if P >= r:
                path = temp_path
            else:
                path = path

        convergence_dist.append(path_distance(path))
        if k%beta == 0:
            T = alpha*T
    
    # print("Convergence", convergence_dist)
    # print("Final Temperature: ", T)
    length = convergence_dist[-1]   
    return path, length, convergence_dist

def main():
    np.random.seed(10) # Seed the random number generator
    p = generate_points()
    np.random.seed(None) # Turn off the random number generator
    
    ########################################

    gpath, glength, _ = greedy_alg(p)
    print("Greedy Algorithm Distance: ", glength)

    # points = np.copy(p)
    # path = np.copy(p)
    # gpath = np.append(path, np.array([points[0]]), axis=0)
    # print("Random Algorithm Distance: ", path_distance(path))

    kmax = 15000
    T = 10
    alpha = 0.95
    beta = 150

    iter = 30
    matrix = []

    fig, axs = plt.subplots(2, 2, figsize=(15, 3))

    # First Condition
    for i in range(iter):
        points = np.copy(p)
        
        path, length, convergence = sim_annealing(gpath, kmax, T, alpha, 250)
        print("Simulated Annealing Distance: ", path_distance(path), f"({i+1}/{iter})")

        matrix.append([path,length,convergence])

    
    matrix.sort(key=lambda x: x[1])
    print("\nSorted Matrix:")
    for item in matrix:
        print(item[1])

    axs[0,0].scatter(points[:, 0], points[:, 1])
    axs[0,0].plot(matrix[round(iter/2)][0][:, 0], matrix[round(iter/2)][0][:, 1], 'r-')
    axs[0,0].plot(0, 0, 'ro')
    axs[0,0].set_xlabel("x1")
    axs[0,0].set_ylabel("x2")
    axs[0,0].set_title("Greedy - Distance: " + str(matrix[round(iter/2)][1]))
    axs[0,0].set_xlim(-55, 55)
    axs[0,0].set_ylim(-55, 55)

    axs[1,0].plot(matrix[round(iter/2)][2])
    axs[1,0].set_xlabel("Iteration")
    axs[1,0].set_ylabel("Distance")
    axs[1,0].set_title("Convergence")

    # Second Condition
    matrix = []
    for i in range(iter):
        points = np.copy(p)
        
        path, length, convergence = sim_annealing(gpath, kmax, T, alpha, 50)
        print("Simulated Annealing Distance: ", path_distance(path), f"({i+1}/{iter})")

        matrix.append([path,length,convergence])

    
    matrix.sort(key=lambda x: x[1])
    print("\nSorted Matrix:")
    for item in matrix:
        print(item[1])

    axs[0,1].scatter(points[:, 0], points[:, 1])
    axs[0,1].plot(matrix[round(iter/2)][0][:, 0], matrix[round(iter/2)][0][:, 1], 'r-')
    axs[0,1].plot(0, 0, 'ro')
    axs[0,1].set_xlabel("x1")
    axs[0,1].set_ylabel("x2")
    axs[0,1].set_title("Greedy - Distance: " + str(matrix[round(iter/2)][1]))
    axs[0,1].set_xlim(-55, 55)
    axs[0,1].set_ylim(-55, 55)

    axs[1,1].plot(matrix[round(iter/2)][2])
    axs[1,1].set_xlabel("Iteration")
    axs[1,1].set_ylabel("Distance")
    axs[1,1].set_title("Convergence")



    #######################################
    # Random Algorithm Implementation
    # points = np.copy(p)
    # path = np.copy(p)
    # gpath = np.append(path, np.array([points[0]]), axis=0)
    # print("\nRandom Algorithm Distance: ", path_distance(path))

    # kmax = 15000
    # T = 10
    # alpha = 0.95
    # beta = 150

    # iter = 5
    # matrix = []

    # for i in range(iter):
    #     points = np.copy(p)
        
    #     path, length, convergence = sim_annealing(gpath, kmax, T, alpha, beta)
    #     print("Simulated Annealing Distance: ", path_distance(path), f"({i+1}/{iter})")

    #     matrix.append([path,length,convergence])

    
    # matrix.sort(key=lambda x: x[1])
    # print("\nSorted Matrix:")
    # for item in matrix:
    #     print(item[1])

    # fig, axs = plt.subplots(2, 2, figsize=(15, 3))
    # axs[0,1].scatter(points[:, 0], points[:, 1])
    # axs[0,1].plot(matrix[0][0][:, 0], matrix[0][0][:, 1], 'r-')
    # axs[0,1].plot(0, 0, 'ro')
    # axs[0,1].set_xlabel("x1")
    # axs[0,1].set_ylabel("x2")
    # axs[0,1].set_title("Random - Distance: " + str(matrix[0][1]))
    # axs[0,1].set_xlim(-55, 55)
    # axs[0,1].set_ylim(-55, 55)

    # axs[1,1].plot(matrix[0][2])
    # axs[1,1].set_xlabel("Iteration")
    # axs[1,1].set_ylabel("Distance")
    # axs[1,1].set_title("Convergence")

    #######################################
    # Plot Histogram

    # gpath, glength, _ = greedy_alg(p)
    # print("Greedy Algorithm Distance: ", glength)

    # kmax = 15000
    # T = 10
    # alpha = 0.95
    # beta = 150

    # iter = 3
    # matrix = []

    # for i in range(iter):
    #     points = np.copy(p)
        
    #     path, length, convergence = sim_annealing(gpath, kmax, T, alpha, beta)
    #     print("Simulated Annealing Distance: ", path_distance(path), f"({i+1}/{iter})")

    #     matrix.append([path,length,convergence])
    
    # matrix.sort(key=lambda x: x[1])
    # print("\nSorted Matrix:")
    # for item in matrix:
    #     print(item[1])

    # for i in range(5,21,5):
    #     plt.figure()
    #     plt.hist([item[1] for item in matrix], bins=i)

    #########################################

    

if __name__ == "__main__":
    main()
    plt.show()