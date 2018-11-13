import numpy as np
import math as mt

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Calculates a set of permutation and associated last changes for TSP
# Input:
#     nNodes: number of nodes in the graph
#     symmetric: whether or not the adjacency matrix is symmetric
#     *args: maximum number of iterations to generate
# Output:
#     l1: list of generated permutations
#     l2: list of indices with the last change

def permute(nNodes, symmetric = False, *args):
    
    if len(args)>0:
        max_iter = min(mt.factorial(nNodes),args[0])
    else:
        max_iter = mt.factorial(nNodes)

    l1 = []
    l2 = []
    lastRedundant = False
    for idx in range(max_iter):
        perm, lastChange = fastPermute(idx, nNodes, symmetric)
        #check if list is empty, i.e. redundant path
        if not perm:
            lastRedundand = True
        else:
            l1.append(perm)
            # If last permutation was skipped, calculate all costs to avoid complications
            if lastRedundant == True:
                l2.append(0)
                lastRedundant = False
            else:
                l2.append(lastChange)    
    return l1, l2

# Just a junction of calculateFactoradic and getPermutation to cut time in half
def fastPermute(num, nNodes, symmetric):
    nodeList = list(range(nNodes))
    perm = []
    lastChange = 0
    for idx, fact in enumerate(np.arange(nNodes)[::-1]):
        hFac = mt.factorial(fact)
        nFac = int(num/hFac)
        perm.append(nodeList[nFac])
        del nodeList[nFac]
        if num > 0 and np.mod(num,hFac) == 0:
            lastChange = idx
        num = np.mod(num,hFac)
    if symmetric == True:
        if perm[0] > perm[-1]:
            perm = []
    return perm, lastChange

# DEPRECATED
# inspiration: https://en.wikipedia.org/wiki/Lehmer_code
# displays num in the factorial number system
def calculateFactoradic(num, nNodes):
    factoradic = []
    for idx in np.arange(nNodes)[::-1]:
        hFac = mt.factorial(idx)
        factoradic.append(int(num/hFac))
        num = np.mod(num,hFac)
    return factoradic

# DEPRECATED
# creates permutation from factoradic
def getPermutation(nNodes, factoradic):
    #this accounts for 
    #if factoradic[0] >= factoradic.count(0):
    #    return []
    #else:
    nodeList = list(range(nNodes))
    perm = []
    for idx in factoradic:
        perm.append(nodeList[idx])
        del nodeList[idx]
    return perm