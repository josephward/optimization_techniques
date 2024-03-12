import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
    

def graph_function(f,n1=100,n2=99,xvalue=10, yvalue=10, xlabel="x1",ylabel="x2"):
    """Generates the graph of the function.

    Parameters:
        f (function):       Objective Function.
        xlabel (string):        Label along x axis
        ylabel (string):        Label along y axis

    Returns:
        Graph of objective function, initial location, and global minimum.
    """

    x1_vect = np.linspace(-xvalue,xvalue,n1)
    x2_vect = np.linspace(-yvalue,yvalue,n2)
    y_vect = np.zeros([n1,n2])

    # Generate the height vector
    for i in range(n1):
        for j in range(n2):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)

    # Plot the curve
    plt.figure("Graph Contour Plot")
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),100,linewidths=2) #Generate Contours
    plt.colorbar()
    plt.show()

def graph_linesearch(f,init_loc,search_points,n1=100,n2=99,xlabel="x1",ylabel="x2"):
    """
    Graphs the result of the linesearch.

    Parameters:
        f (function):           Objective function.
        init_loc (list):        List of coordinates of initial condition.
        search_points (list):   List of points arrived at during the linesearch
        
        n1 (int):               X resolution of graph
        n2 (int):               Y resolution of graph
        xlabel (string):        Label along x axis
        ylabel (string):        Label along y axis

    Returns:
        Graph of linesearch with each substep

    """
    
    # Graph down to the dimensions of the linesearch 
    maxx = 0
    minx = 999
    maxy = 0
    miny = 999
    for i in range(len(search_points)):
        tempx = search_points[i][0]
        tempy = search_points[i][1]
        if (tempx > maxx):
            maxx = round(tempx,2)
        if (tempx < minx):
            minx = round(tempx,2)
        if (tempy > maxy):
            maxy = round(tempy,2)
        if (tempy < miny):
            miny = round(tempy,2)

    # Generate the graphing space
    spacing_var = 0.1*(maxx-minx)
    x1_vect = np.linspace(round(minx,2)-spacing_var,round(maxx,2)+spacing_var,n1)
    x2_vect = np.linspace(round(miny,2)-spacing_var,round(maxy,2)+spacing_var,n2)
    y_vect = np.zeros([n1,n2])

    # Generate the height vector
    for i in range(n1):
        for j in range(n2):
            x = [x1_vect[i],x2_vect[j]]
            y_vect[i,j] = f(x)

    # Plot the curve
    CS = plt.contour(x1_vect,x2_vect,np.transpose(y_vect),25,linewidths=2) #Generate Contours
    plt.clabel(CS, inline=True, fontsize=10)

    # Plot global min point
    res = minimize(f,init_loc) # Find minimum
    print("Actual Min: ",res.x)
    plt.plot(res.x[0],res.x[1],"r*") #Plot minimum
    plt.annotate("Scipy Min",[res.x[0],res.x[1]])

    # Plot linesearch values
    plt.plot()
    for i in range(len(search_points)-1):
        plt.plot([search_points[i][0],search_points[i+1][0]],[search_points[i][1],search_points[i+1][1]],"ro-")

    #Annotate Graph
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def plot_convergence(f_grad,points,desc_text="Convergence Plot",xlabel="Iterations",ylabel="Norm of Gradient"):
    """
    Plots the convergence of the optimization method over time. Norm of gradient vs time.
    
    Parameters:
        f_grad (function):  Function that returns the gradient of a function.
        desc_text (string): Window title
        xlabel (string):    Label on the x axis
        ylabel (string):    Label on the y axis
    
    Returns:
        Convergence graph.
    """
    plt.figure(desc_text)
    
    x_vect = np.arange(0,len(points),1)
    y_vect = []
    for x in range(len(x_vect)):
        y_vect.append(np.linalg.norm(f_grad(points[x])))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Check if should be log scale
    vlength = round(len(y_vect)/6)
    print(vlength)
    print(y_vect[vlength],y_vect[0])
    if (np.abs(y_vect[0]-y_vect[vlength])>100):
        plt.yscale('log')

    plt.plot(x_vect,y_vect)