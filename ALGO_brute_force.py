#!/usr/bin/env python
# coding: utf-8

__author__ = "Eduardo Brandao"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# In[3]:


import graph_class as g
import itertools
import numpy as np
import time


# In[4]:


#brute force with a random start implementation; set max_iterations if the graph has a more than 10 nodes!
#returns elapsed time, the minimum distance, and the minimum cycle/plot of the cycle
def brute_force(graph, max_iterations = np.infty, random_init = True, return_graph = True):
    """
    graph: instance of class graph; can be created from coordinate list (graph.fully_connected_graph_from_coordinate_list),
    from an adjacency matrix (graph.graph_from_adjacency_matrix) or from a tsp type file (graph.heidelberg_2D)
    max iterations: DO use if graph has more than 10 nodes!
    random_init: set to True, starts from a random cycle; otherwise it starts from implementation sequence cycle
    return_graph: set to True, displays a plot of the graph; otherwise, it returns the vertices (class vertex objects)
    """
    start = time.time()    
    
    if random_init:
        nodes = np.random.permutation(graph.vertices)
    else:
        nodes = graph.vertices
        
    #creates an iterator instead of saving all paths (which is impossible except for very small problems)      
    allpaths = iter((itertools.permutations(nodes)))
    min_weight = np.infty
    step = 0
    
    while step < max_iterations:
        step += 1
        this_path = next(allpaths)
        
        current_path_weight = graph.get_cycle_weight(this_path)
        
        if current_path_weight < min_weight:
            min_weight = current_path_weight
            min_path = this_path
            
    end = time.time()
    if return_graph:
        node_output = g.plotTSP_2D(min_path)
    else:
        node_output = min_path
    return end-start, min_weight, node_output


# In[ ]:




