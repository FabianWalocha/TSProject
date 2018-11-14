#!/usr/bin/env python
# coding: utf-8

# In[1]:


import graph_class as g
import itertools
import numpy as np
import time


# In[2]:

#returns elapsed time, the minimum distance, and the minimum cycle/plot of the cycle
def simulated_annealing(graph,random_init = True, temperature = 100000, return_graph = True):
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
    #2. Instead of shuffling a random path, should maximize the distance between the two permutations (since the length 
        # of the sequence to be shuffled is already random). This could be done using, perhaps, the "Kendal tau distance":
        # [[https://en.wikipedia.org/wiki/Kendall_tau_distance]]
    start = time.time()
    if random_init:
        nodes = np.random.permutation(graph.vertices)
    else:
        nodes = graph.vertices
        
    current_min = graph.get_cycle_weight(nodes)
    
    for i in range(temperature):
        k = np.random.randint(0,len(nodes))
        p = np.random.randint(k,len(nodes))
        np.random.shuffle(nodes[k:p])
        test_min = graph.get_cycle_weight(nodes)
        # I chose a Gaussian but other probability distributions should also make sense
        if np.exp(-(test_min-current_min)**2/(temperature-i))> np.random.rand():
            current_min = test_min
        else:
            current_min = min(current_min, test_min)    
    
    end = time.time()
    if return_graph:
        node_output = g.plotTSP_2D(nodes)
    else:
        node_output = nodes
    return end-start, current_min, node_output

