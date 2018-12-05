import sys
sys.path.insert(0, '../../TSProject/')

import numpy as np
import random as rd
from time import time
import multiprocessing

__author__ = "Fabian Walocha"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# Implementation of the ant colony algorithm for TSP
# Input:
#     adj_mat: adjacency matrix
#     num_agents: how many ants are supposed to traverse paths at the same time
#     max_iter: how many times num_agents ants are supposed to be send to traverse
#     alpha, beta: parameters which influence the effect size of the pheromone trail and the path length on the probabilistic update
#     rho: multiplier which regulates the decay rate of the pheromone trail
#     verbose: whether you want print updates every 1000 iterations on the probability distribution of the 
# Output:
#     cost: cost found
#     path: associated path
#     prob_mat: probability matrix obtained in the end (rounded on 2 digits)
#     time: time it took for the algorithm to finish

# inspiration: http://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
# parameter optimization: https://pdfs.semanticscholar.org/30cc/a5e0f5f84d3ad4c4224e247cb418304b4721.pdf
def antColonyOptimization(graph, num_agents=20, max_iter = 1000, alpha=1, beta=5, rho=0.5, symmetric=True, verbose=0):
    adj_mat = graph.weighted_adjacency_matrix
    t1 = time()
    cmin = np.inf
    pmin = []
    pher_mat = np.ones(adj_mat.shape)
#     prob_mat = np.ones(adj_mat.shape)/len(adj_mat-1)
    prob_mat = np.ones(adj_mat.shape)/(len(adj_mat)-1)
    for idx in range(len(adj_mat)):
        prob_mat[idx,idx]=0
    idx = 0
    tdif = 0
    while idx<max_iter and tdif<600:
        if verbose > 0:
            if np.mod(idx,1000)==0:
                print("Iteration no.",idx,"Result",np.min(np.max(prob_mat,axis=1)))
        # Get random starting points
        positions = rd.choices(range(len(adj_mat)),k=num_agents)
        # find a path with each ant, decay pheromones, update pheromones
        pher_mat, cmin, pmin = traverse_random_multi(adj_mat, prob_mat, pher_mat, cmin, pmin,num_agents, positions, rho, symmetric)
        # Update the probabilities given the new pheromone trails
        prob_mat = update_probs(adj_mat, pher_mat, alpha, beta)
        idx += 1
        ttemp = time()
        tdif = (ttemp-t1)
        
    t2 = time()
    return cmin, pmin, (t2-t1)
    
# sends multiple ants to find a path, updates pheromone trail
def traverse_random_multi(adj_mat, prob_mat, pher_mat, cmin, pmin, num_agents, positions, rho, symmetric):
    results = []
    for idx in range(num_agents):
        path, pheromones, cmin, pmin = traverse_random(adj_mat, prob_mat, positions[idx], cmin, pmin)
        results.append([path, pheromones])
    # Decay the pheromone trail from before
    pher_mat = rho*pher_mat
    # Update the pheromone trail with new paths
    for idx in range(len(results)):
        # if route could not be created
        if not results[idx][0]:
            continue
        for idx2 in range(len(adj_mat)):
            pher_mat[results[idx][0][idx2],results[idx][0][idx2+1]]+=results[idx][1]
            if symmetric:
                pher_mat[results[idx][0][idx2+1],results[idx][0][idx2] ]+=results[idx][1]
    return pher_mat, cmin, pmin
    
# send a single ant to find a path
def traverse_random(adj_mat, prob_mat, position, cmin, pmin):
    initial_position = position
    # Memory of the ants -> tabu list
    path = [initial_position]
    pheromones = 0
    p_mat = np.copy(prob_mat)
    for idx in range(len(adj_mat)-1):
        try:
            probs = np.cumsum(p_mat[position,:])
            probs = probs/probs[-1]
            goal_position = [i for i, x in enumerate(probs-rd.random()) if x>0][0]
        except IndexError: 
            return [], [], cmin, pmin
        for p in path:
            p_mat[goal_position,p] = 0.0
        path.append(goal_position)
        # update function depending of length of the edge
        pheromones += adj_mat[position,goal_position]
        position = goal_position
    pheromones += adj_mat[position,initial_position]
    path.append(initial_position)
    if pheromones < cmin:
        cmin = pheromones
        pmin = path
    pheromones = 1./pheromones
    return path, pheromones, cmin, pmin

# update the probability given the distances and the new pheromone trails
def update_probs(adj_mat, pher_mat, alpha, beta):
    prob_mat = np.zeros(adj_mat.shape)
    for idx in range(len(adj_mat)):
#         tau = np.array([np.min([1,1./x]) for x in 1./adj_mat[idx,:]])
        normalization = np.sum(pher_mat[idx,:]**alpha * (1./adj_mat[idx,:])**beta)
        prob_mat[idx,:] =   (pher_mat[idx,:]**alpha * (1./adj_mat[idx,:])**beta) / normalization
    return prob_mat