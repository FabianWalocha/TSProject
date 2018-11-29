import sys
sys.path.insert(0, '../../TSProject/')

import numpy as np
import random as rd
from time import time 

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Implementation of the greedy algorithm for TSP
# Input:
#     adj_mat: adjacency matrix
# Output:
#     cost: cost found
#     list_of_traversed_nodes: associated path
#     time: time needed to execute the algorithm

def greedy(graph):
    adj_mat = graph.weighted_adjacency_matrix
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