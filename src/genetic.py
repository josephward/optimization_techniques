"""
Implementation of the Genetic Algorithm with a test case.

By Joseph Ward March 9th 2024
"""

import numpy as np


def egg_carton(x):
    """Example function to demonstrate function"""
    return 0.1*x[0]**2+0.1*x[1]**2-np.cos(3*x[0])-np.cos(3*x[1])

def genetic_algorithm():
    """Implementation of the Genetic Algorithm algorithm"""
    pass