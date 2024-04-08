"""
Implementation of the knapsack algorithm
"""

import numpy as np

def knapsack(matrix, K):
    """
    matrix (2d vector)  = weights and values, ex [[weight, value]...]
    capacity (int)      = the maximum weight that can be carried
    """
    K += 1
    length = len(matrix[:, 0])
    # V and S are length x K sized
    # Initialize with zeros
    v = np.zeros((K, length))
    s = np.zeros((K, length))

    # Remember matrix[y][x]
    for i in range(1, length):  # For every option
        w = matrix[i][0] # Weight of object
        c = matrix[i][1] # Value of object
        for k in range(K):      # For every weight
            if w > k:
                v[k][i] = v[k][i-1]
            else:
                if c + v[k-w][i-1] > v[k, i-1]:
                    v[k][i] = c + v[k-w][i-1]
                    s[k][i] = 1
                else:
                    v[k][i] = v[k][i-1]
        # print("Items:",i,"Weight:", k)

    # Collect Solution
    k = K
    xstar = []
    locstar = []
    # print("Len", length)
    # print("k",k)
    # print(s.shape)
    for i in range(length-1, -1, -1):
        w = matrix[i][0] # Weight of object
        if s[k-1][i] == 1:
            xstar.append(matrix[i][1])
            locstar.append(i+1)
            k -= w
    return v, s, xstar, locstar

def main():
    capacity = 40
    
    matrix = np.array([[7,16000],[4,1000],[2,7000],[2,14000],[3,6000],[4,10000],[7,12000],[5,18000],[4,5000],[3,11000],[2,6000],[3,13000],[2,18000],[2,5000],[2,17000],[7,17000],[3,7000],[2,12000],[1,14000],[4,9000],[4,9000],[1,6000],[4,3000],[7,6000],[6,12000],[7,8000],[2,19000]])
    length = len(matrix[:, 0])
    
    # Greedy Solution
    # greedy_matrix = sorted(matrix, key=lambda x: x[1]/x[0], reverse=True)
    # greedy_matrix = sorted(enumerate(matrix), key=lambda x: x[1][1]/x[1][0], reverse=True)

    # greedy_value = 0
    # K = capacity
    # loc = []
    # loc2 = []
    # for i in range(length):
    #     greedy_value += greedy_matrix[i][1][1]
    #     K -= greedy_matrix[i][1][0]
    #     loc.append(greedy_matrix[i][1][1])
    #     loc2.append(greedy_matrix[i][0] + 1)
    #     if K < 0:
    #         greedy_value -= greedy_matrix[i][1][1]
    #         K += greedy_matrix[i][1][0]
    #         loc.pop()
    #         loc2.pop()
    #         break
    # print(K)
    # print(f"Greedy Value: ${greedy_value}")
    # print(loc)
    # print(sorted(loc2, reverse=True))

    # # Greedy Solution 2
    # K = capacity + 1
    # bag = np.array([])
    # value = 0
    # flag = 0
    # for i in range(max(matrix[:][0])):
    #     if flag == 1:
    #         break
    #     for j in range(len(matrix[:, 0])):
    #         print(K)
    #         if matrix[j][0] == i:
    #             if K - matrix[j][0] < 0:
    #                 flag = 1
    #                 break
    #             bag = np.append(bag, matrix[j][1])
    #             value += matrix[j][1]
    #             K -= matrix[j][0]
    
    # print(bag)
    # print(value)
    # print(K)

    # Greedy Solution 3
    K = capacity + 1
    bag = np.array([])
    value = 0
    while K > 0:
        for i in range(len(matrix[:, 0])):
            if K - matrix[i][0] < 0:
                print(K)
                K = -10
                break
            else:
                bag = np.append(bag, matrix[i][1])
                value += matrix[i][1]
                K -= matrix[i][0] 
    print(value)
    print(bag)
    print(K)

    # # Knapsack Solution
    # v, s, xstar, locstar = knapsack(matrix, capacity)
    # value = sum(xstar)
    # print(f"Knapsack Value: ${value}")
    # # print(s)
    # # print(v)
    # print("Xstar\n",xstar)
    # print("Locstar\n",locstar)

    # # Print the V Matrix
    # for i in range(capacity+1):
    #     print(', '.join(map(lambda x: str(int(x)), v[i, :])))
    #     pass
    # Print the S matrix 
    # for i in range(capacity+1):
    #     print(', '.join(map(lambda x: str(int(x)), s[i, :])))
    #     pass

    

if __name__ == "__main__":
    main()