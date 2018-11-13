import permute as pm
import numpy as np
from time import time

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Implementation of the brute force algorithm for TSP
# Input:
#     adj_mat: adjacency matrix
#     symmetric: wheter or not the symmetric property can be used
#     timed: whether or not the algorithm is supposed to be timed
#     *args: maximum number of iterations to generate
# Output:
#     cMin: minimum cost found
#     pMin: associated minimum path
#     time: time needed to execute the algorithm (if timed=True)

def bruteForce(adj_mat, symmetric=False, timed = False, *args):
    
    # create the permutations for the algorithm
    # this part is not timed since it does not depend on the input adjacency matrix
    nNodes = len(adj_mat) -1
    if len(args)>0:
        permList = pm.permute(nNodes,symmetric,args[0])
    else:
        permList = pm.permute(nNodes,symmetric)
    perms = np.array(permList[0])+1
    lastChanges = permList[1]
    
    if timed:
        t1 = time()
    
    cMin = np.inf
    # used for storing previous costs to not compute them again
    cLast = np.zeros([nNodes-1,1])
    pMin = []
    for perm_idx, p in enumerate(perms):
        if lastChanges[perm_idx] == 0:
            cCurr = adj_mat[0,p[0]]
            cLast[0] = cCurr
        # reuse previous calculations if possible to speed up the process
        else:
            cCurr = cLast[lastChanges[perm_idx]-1]
        for idx in range(lastChanges[perm_idx],len(p)-1):
            cCurr = cCurr+adj_mat[p[idx],p[idx+1]]
            cLast[idx] = cCurr
        cCurr = cCurr + adj_mat[p[-1],0]
        if cCurr < cMin:
            pMin = p
            cMin = cCurr
        
    if timed:    
        t2 = time()
        return cMin, np.append([0],np.append(pMin,[0])), (t2-t1)
    else:
        return cMin, np.append([0],np.append(pMin,[0]))