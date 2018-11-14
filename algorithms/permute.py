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
            if not lastRedundant:
                # save the path which was the same for the last (redundant) path
                lastPos = lastChange
                lastRedundant = True
            else:
                # update the path which was the same for each new (redundant) path
                lastPos = np.min([lastPos,lastChange])
        else:
            l1.append(perm)
            # If last permutation(s) were skipped, calculate from last common path
            if lastRedundant:
                l2.append(np.min([lastPos,lastChange]))
                lastRedundant = False
            else:
                l2.append(lastChange)    
    return l1, l2

# Just a junction of calculateFactoradic and getPermutation to cut time in half
def fastPermute(num, nNodes, symmetric):
    nodeList = list(range(nNodes))
    perm = []
    lastChange = 0
    # Get each digit of factoradic which is in base mt.factorial(fact)
    for idx, fact in enumerate(np.arange(nNodes)[::-1]):
        # get base
        hFac = mt.factorial(fact)
        # how often does it fit in base -> nFac
        nFac = int(num/hFac)
        perm.append(nodeList[nFac])
        # update permutation by deleting the nFac-th element from list
        del nodeList[nFac]
        if num > 0 and np.mod(num,hFac) == 0:
            lastChange = idx
        num = np.mod(num,hFac)
    # delete all redundant paths
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