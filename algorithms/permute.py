import numpy as np
import math as mt

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# inspiration: https://en.wikipedia.org/wiki/Factorial_number_system#Permutations
# Calculates a set of permutation
# Input:
#     nNodes: number of nodes in the graph
#     *args: maximum number of iterations to generate (default is nNodes!)
# Output:
#     l1: list of generated permutations
#     l2: list of indices with the last change
def permute(nNodes,*args):
    if len(args)>0:
        max_iter = min(args[0],mt.factorial(nNodes))
    else:
        max_iter = mt.factorial(nNodes)
    l1 = []
    l2 = []
    for idx in range(max_iter):
        perm, lastChange = fastPermute(idx, nNodes, args)
        l1.append(perm)
        l2.append(lastChange)
    return l1, l2

def fastPermute(num, nNodes, *args):
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
    return perm, lastChange

# DEPRECATED
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