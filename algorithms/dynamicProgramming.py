import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (DynamicProgramming).
def DynamicProgramming(graph, timed = False):
    
    adj_matrix = graph.weighted_adjacency_matrix
    
    M = np.matrix(adj_matrix)
    
    for i in range(0,np.size(M,1)):
        M[i,i] = np.inf       
    
    n = np.size(M,1)

    L = list(range(2,n+1))

    full_set = tuple([1,tuple(range(2,len(L)+2))])
    S = list(itertools.combinations(L,1))
    for i in range(1,len(L)+1):
        for j in L:
            exclude = [] + L
            exclude.remove(j)
            combos = list(itertools.combinations(exclude, i))
            S += [tuple([j,x]) for x in combos]
    S.append(full_set)
            
    # start algorithm
    cost_memo = {}
    path_memo = {}

    t1 = time.time()

    #We compute the edges from the leaves to the starting point
    for k in S[0:len(L)]:
        cost_memo[k] = M[k[0]-1,0]
        path_memo[k] = [k[0]]

    # We traverse through all the list of subsets
    for k in S[len(L):]:
        
        # Initialize minimal cost
        min_cost = np.inf 
        
        # Basic cases
        if len(k[1]) == 1:
            s = tuple(k[1])
            
            # YOU CAN CHANGE THE MAXIMUM TIME HERE
            if timed:
                if time.time()-t1 > 600:
                    return cost_memo[s], path_memo[s], time.time() - t1  

            cost_memo[k] = M[k[0]-1,s[0]-1] + cost_memo[s]
            path_memo[k] = [k[0]] + path_memo[s]
            
            continue
        
        # Initialize min cost
        min_cost = np.inf
        
        # We traverse through the elements of the permutation
        for j in k[1]: # This index removes the element corresponding to the node that we will visit

            # We rewrite the available paths as an already solved problems
            avail_nodes = list(k[1])
            avail_nodes.remove(j)
            s = tuple([j,tuple(avail_nodes)])
            
            # Compute the cost
            cost = M[k[0]-1,j-1] + cost_memo[s]
            
            # YOU CAN CHANGE THE MAXIMUM TIME HERE
            if timed:
                if time.time()-t1 > 600:
                    return cost_memo[s], path_memo[s], time.time() - t1  

            if cost < min_cost:
                min_cost = cost
                min_path = [k[0]] + path_memo[s]
        
        # We save in our memos the cost and its associated path
        cost_memo[k] = min_cost    
        path_memo[k] = min_path

    # We add the set corresponding to (1,2,...,n) to our calculations


    path_memo[full_set] = min_path + [1]
    cost_memo[full_set] = min_cost
    
    return cost_memo[full_set], path_memo[full_set], time.time() - t1