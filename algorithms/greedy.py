import sys
sys.path.insert(0, '../../TSProject/')

import numpy as np
import random as rd
from time import time 

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
    nNodes = len(adj_mat)
    firstMove = np.unravel_index(np.argmin(adj_mat),adj_mat.shape)
    listOfTraversedNodes = [firstMove[0],firstMove[1]]
    cost = adj_mat[firstMove[0],firstMove[1]]
    while len(listOfTraversedNodes) < len(adj_mat):
        currentCosts = list(adj_mat[listOfTraversedNodes[-1],:])
        for idx2 in listOfTraversedNodes:
            currentCosts[idx2] = np.inf
        listOfTraversedNodes.append(np.argmin(currentCosts))
        cost += adj_mat[listOfTraversedNodes[-2],listOfTraversedNodes[-1]]
    cost += adj_mat[listOfTraversedNodes[-1],listOfTraversedNodes[0]]
    t2 = time()
    return cost, listOfTraversedNodes + [listOfTraversedNodes[0]], (t2-t1)