import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time

def bruteForce(coordinates):

    # Convert to matrix
    coordinates = np.matrix(coordinates)

    # The number of verteces 
    n = np.size(coordinates,0)

    # Adjacency matrix
    # NOT OPTIMAL GENERATION, JUST FOR TESTING
    M = np.matrix([[int(np.around(np.linalg.norm(coordinates[i,0:2]-coordinates[j,0:2]))) for j in range(n)] for i in range(n)])

    # Generate permutations of n
    # NOTE: THERE ARE DUPLICATE PATHS
    solutionSpace = list(permutations(range(n)))

    bestDistance = np.inf
    bestPath = tuple(range(1,n+1))

    t1 = time.time()

    for p in solutionSpace:
        
        dist = 0
        for i in range(n-1):
            dist += M[p[i],p[i+1]]
        dist += M[p[n-1],p[0]]
        
        if dist < bestDistance:
            bestDistance = dist
            bestPath = p

    t2 = time.time()

    return (t2-t1, bestDistance)
