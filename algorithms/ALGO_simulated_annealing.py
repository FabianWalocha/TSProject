#!/usr/bin/env python
# coding: utf-8

# In[1]:

__author__ = "Eduardo Brandao"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import graph_class as g
import itertools
import numpy as np
import time


# In[2]:

#returns elapsed time, the minimum distance, and the minimum cycle/plot of the cycle
def simulated_annealing(graph,random_init = True, temperature = 10000, cooling_factor = 0.999, return_graph = False, method = "2-OPT", break_time = 600):
    """"
    graph: instance of class graph; can be created from coordinate list (graph.fully_connected_graph_from_coordinate_list),
    from an adjacency matrix (graph.graph_from_adjacency_matrix) or from a tsp type file (graph.heidelberg_2D)
    random_init: set to True, starts from a random cycle; otherwise it starts from implementation sequence cycle
    temperature: initial "temperature" of the graph. The greater it is, the greater the possibility of an inferior cycle 
    being accepted
    return_graph: set to True, displays a plot of the graph; otherwise, it returns the vertices (class vertex objects)
    """
    #ISSUES:
    #1. Cools one degree for every computation step. Should be decoupled: simulate cooling time using a "cooling function"
    # DONE, using Kirpatrick cooling
    #2. Instead of shuffling a random path, should maximize the distance between the two permutations (since the length 
        # of the sequence to be shuffled is already random). This could be done using, perhaps, the "Kendal tau distance":
        # [[https://en.wikipedia.org/wiki/Kendall_tau_distance]]
    # DONE: there's no such thing; I just chose another neighbouring method (2-OPT), which is faster to compute
    
    start = time.time()
    
    if random_init:
        nodes = np.random.permutation(graph.vertices)
    else:
        nodes = graph.vertices
        
    current_min = graph.get_cycle_weight(nodes)
    
    # with default temperature and cooling rate, it will go through about 23000 iterations,
    while temperature>0.000001:
        
        k = np.random.randint(0,len(nodes))
        p = np.random.randint(k,len(nodes))
        
        # this is the NEIGHBOURING algorithm in the pseudocode        
        # we can choose to expore the problem space by either
        # doing a random SHUFFLE of the nodes between k and p or a full inversion (2-OPT)
        # 2-OPT should be faster to compute, but the set of neighbours is smaller at each iteration
        # there are paths in solution space that are not accessible to 2-OPT, which are accessible to SHUFFLE;
        # there should be some classes of problems for which each method converges faster
        
        if method == "SHUFFLE":
            np.random.shuffle(nodes[k:p])
        elif method == "2-OPT":    
            nodes[k:p]= np.flip(nodes[k:p], axis=0)
        
        test_min = graph.get_cycle_weight(nodes)
        
        # Replaced Gaussian with Boltzmann distribution, in the spirit of good physics ;)
        if np.exp(-(test_min-current_min)/(temperature))> np.random.rand():
            current_min = test_min
        else:
            current_min = min(current_min, test_min)
        if time.time()-start > break_time:
            break
        # replaced linear cooling with kirpatrick cooling(http://citeseer.ist.psu.edu/kirkpatrick83optimization.html),
        # again in the spirit of good physics and because we want the algorithm to spend enough time perfecting the 
        # local minimum
        temperature = temperature*cooling_factor
    
    end = time.time()
    if return_graph:
        node_output = g.plotTSP_2D(nodes)
    else:
        node_output = nodes
    return end-start, current_min, node_output
