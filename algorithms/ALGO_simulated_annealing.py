import graph_class as g
import itertools
import numpy as np
import time
import random as rd
from matplotlib import pyplot as plt

def greedy_just_for_sa(graph):
    """"
    Thanks Fabian!
    """"
    adj_mat = graph.weighted_adjacency_matrix
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
    return listOfTraversedNodes



def simulated_annealing(graph, iterations=0, greedy_start = True, temperature=10,
           cooling_factor = 0.99999, return_graph = False, method = "2-OPT", break_time = 600):
    
    start = time.time()
    
    if greedy_start:
        nodes = [graph.vertices[index] for index in greedy_just_for_sa(graph)]
    else:
        nodes = np.random.permutation(graph.vertices)
    
    
    current_min = graph.get_cycle_weight(nodes)
    
    while temperature>1:
        #difference in paths can't be greater than one
        #for very small T, e^{-\DeltaE/T} is going accept small \Delta E with great likelihood!

            #generate random nodes 
            k = np.random.randint(0,len(nodes))
            p = np.random.randint(k,len(nodes))

            test_nodes = nodes.copy()
            
            if method == "SHUFFLE":
                np.random.shuffle(test_nodes[k:p])
            elif method == "2-OPT":
                test_nodes[k:p] = np.flip(test_nodes[k:p], axis=0)

            current_path_length = graph.get_cycle_weight(nodes)
            test_path_length = graph.get_cycle_weight(test_nodes)

            # Boltzmann distribution
            Delta_Length = test_path_length - current_path_length

            p = min(np.exp(-(Delta_Length)/(temperature)),1)
            
            # testing a few hundred runs shows that <= is marginally better on average,
            # but better at finding minima, since variance tends to be greater
            # I believe this happens because always accepting same-length configurations
            # allows it to explore the solution space better
            if Delta_Length <= 0:
                nodes = test_nodes.copy()

            elif p > np.random.rand():
                    nodes = test_nodes.copy()
            #kirpatrick cooling(http://citeseer.ist.psu.edu/kirkpatrick83optimization.html)
            temperature = temperature*cooling_factor
            if iterations>0:
                if temperature <1:
                    temperature = 100
                    iterations-=1
            
            if time.time()-start > break_time:
                break
    end = time.time()
    if return_graph:
        node_output = g.plotTSP_2D(nodes)
    else:
        node_output = nodes
    return end-start, graph.get_cycle_weight(nodes), node_output
