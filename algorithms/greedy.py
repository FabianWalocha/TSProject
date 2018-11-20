import numpy as np
import random as rd
from time import time 

def greedy(adj_mat, timed = True):
    if timed:
        t1 = time()
    cost = 0
    listOfTraversedNodes = []
    nNodes = len(adj_mat)
    listOfTraversedNodes.append(np.unravel_index(np.argmin(adj_mat),adj_mat.shape)[0])
    for idx in range(nNodes-1):
        currentCosts = list(adj_mat[idx,:])
        for idx2 in listOfTraversedNodes:
            currentCosts[idx2] = np.inf
        listOfTraversedNodes.append(np.argmin(currentCosts))
        cost = cost + adj_mat[listOfTraversedNodes[idx],listOfTraversedNodes[idx+1]]
    cost = cost+ adj_mat[listOfTraversedNodes[-1],listOfTraversedNodes[0]]
    t2 = time()
    return cost, listOfTraversedNodes + [listOfTraversedNodes[0]], (t2-t1)
            