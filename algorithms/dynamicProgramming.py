import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (DynamicProgramming).
def DynamicProgramming(graph, timed = False):
    
    adj_mat = graph.weighted_adjacency_matrix
    adj_mat = np.matrix(adj_mat)
    
    n = np.size(adj_mat,1)

    L = list(range(2,n+1))

    all_S = []
    for i in range(1,len(L)+1):
        all_S = all_S + list(itertools.combinations(L, i))
    
    
    for i in range(0,n):
        adj_mat[i,i] = np.inf    
    
    # start algorithm
    cost_memo = {}
    path_memo = {}
    
    #We compute the edges from the leaves to the starting point
    for k in all_S[0:len(L)]:
       cost_memo[k] = M[k[0]-1,0]
       path_memo[k] = [k[0]]

    t1 = time.time()
    
    # We traverse through all the list of subsets
    for k in all_S[len(L):]:

        # Initialize minimal cost
        min_cost = np.inf 

        # We traverse through the elements of the permutation
        for i in range(0,len(k)): # This index removes the element corresponding to the node that we will visit
            for j in range(0,len(k)): # This index corresponds to the current node, i != j
                                      # E.G. [k={2,3,4}, i=3, j=2] ==> [2 -> 3, with j in k-{3}] 
                    
                # YOU CAN CHANGE THE MAXIMUM TIME HERE
                if timed:
                    if time.time()-t1 > 600:
                        return cost,path_memo[s], time.time() - t1                   
                    
                if i==j: continue

                # We remove "i" from the subset, but tuples are immutable
                s = list(k)
                s.remove(k[i])
                s = tuple(s)

                # The new cost is the sum of the cost of traveling to i + accumulated cost
                cost = M[k[i]-1,k[j]-1] + cost_memo[s]


                # We look for minimum cost and its path
                if cost < min_cost:
                    min_cost = cost
                    min_path = path_memo[s] + [k[i]]

        # We save in our memos the cost and its associated path
        cost_memo[k] = min_cost    
        path_memo[k] = min_path

    # We add the set corresponding to (1,2,...,n) to our calculations
    full_set = tuple(range(1,len(L)+2))
    path_memo[full_set] = [1] + min_path
    cost_memo[full_set] = min_cost + M[0,min_path[-1]-1]

    
    return cost_memo[full_set], path_memo[full_set], time.time() - t1