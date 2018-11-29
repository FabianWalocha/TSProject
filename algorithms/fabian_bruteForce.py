import sys
sys.path.insert(0, '../../TSProject/')

from algorithms import permute as pm
import numpy as np
from time import time
import itertools
import math as mt

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Implementation of the brute force algorithm for TSP
# Input:
#     adj_mat: adjacency matrix
#     symmetric: wheter or not the symmetric property can be used
#     preload: preload permutations (decreases runtime, recommended for graphs <11 nodes only!)
#     *args: maximum number of iterations to generate
# Output:
#     cMin: minimum cost found
#     pMin: associated minimum path
#     time: time needed to execute the algorithm

def bruteForce(graph, symmetric=False, preload=False, *args):
    adj_mat = graph.weighted_adjacency_matrix
    
    # create the permutations for the algorithm
    nNodes = len(adj_mat) -1
    
    # this part is not timed since it does not depend on the input adjacency matrix
    if preload:
        if len(args)>0:
            permList = pm.permute(nNodes,symmetric,args[0])
        else:
            permList = pm.permute(nNodes,symmetric)
        perms = np.array(permList[0])+1
        lastChanges = np.array(permList[1])+1
        
    t1 = time()
    
    cMin = np.inf
    # used for storing previous costs to not compute them again
    cLast = np.zeros([nNodes,1])
    pMin = []
    
    if len(args)>0:
        max_iter = np.min([args[0],mt.factorial(nNodes)])
    else:
        max_iter = mt.factorial(nNodes)
    
    if preload:
        for perm_idx, p in enumerate(perms):
            # reuse previous calculations when possible to speed up the process
            if lastChanges[perm_idx] > 1:
                cCurr = cLast[lastChanges[perm_idx]-2]
            for idx in range(lastChanges[perm_idx]-1,len(p)):
                # If no usable calculations, start from scratch
                if idx == 0:
                    cCurr = adj_mat[0,p[0]]
                    cLast[0] = cCurr
                # For each further one, go through permutations one by one
                else:
                    cCurr = cCurr+adj_mat[p[idx-1],p[idx]]
                    cLast[idx] = cCurr
            # Add the way to the first element at the end
            cCurr = cCurr + adj_mat[p[-1],0]
            if cCurr < cMin:
                pMin = np.append([0],np.append(p,[0]))
                cMin = cCurr
            ttemp = time()
            if (ttemp-t1) > 600:
                break
    else:
        perms = iter((itertools.permutations(range(nNodes))))
        for perm_idx in range(max_iter):            
            p = np.array(next(perms))+1
            if perm_idx == 0:
                lastChanges = 1
                lastRedundant = False
            else:
                if lastRedundant:
                    lastChanges = np.min([lastChanges,[i for i,x in enumerate(plast!=p) if x][0]+1])
                else:
                    lastChanges = [i for i,x in enumerate(plast==p) if x][0]+1
            if p[0]>p[-1]:
                if lastChanges > 1:
                    cCurr = cLast[lastChanges-2]
                for idx in range(lastChanges-1,len(p)):
                    # If no usable calculations, start from scratch
                    if idx == 0:
                        cCurr = adj_mat[0,p[0]]
                        cLast[0] = cCurr
                    # For each further one, go through permutations one by one
                    else:
                        cCurr = cCurr+adj_mat[p[idx-1],p[idx]]
                        cLast[idx] = cCurr
                # Add the way to the first element at the end
                cCurr = cCurr + adj_mat[p[-1],0]
                if cCurr < cMin:
                    pMin = np.append([0],np.append(p,[0]))
                    cMin = cCurr
            else:
                lastRedundant = True            
            plast = p
            ttemp = time()
            if (ttemp-t1) > 600:
                break
                
    # If more than one minimal solution was found in between
    if isinstance(cMin, np.ndarray):
        cMin = cMin[0]
    t2 = time()
    return cMin, pMin, (t2-t1)