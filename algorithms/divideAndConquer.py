import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (DivideAndConquer).
def DivideAndConquer(graph, timed = False):
    
    adj_mat = graph.weighted_adjacency_matrix
    adj_mat = np.matrix(adj_mat)
    
    n = np.size(adj_mat,1)

    for i in range(0,n):
        adj_mat[i,i] = np.inf

    L = list(range(2,n+1))
    
    t1 = time.time()
    
    cost, path, duration = DnC(1,L,timed, t1)
    
    return cost, path, duration

def DnC(i,rem_nodes, timed, t1):
    
    # If there are no more nodes to traverse, it means we reached the end of the path, so we go back to the initial vertex
    if len(rem_nodes) == 0:
        # Calculates the distance between the node and the starting point
        cost = M[i -1,0]
        return cost, [1,i], time.time()-t1
    
    else:
        
        # Initialize minimum cost
        min_cost = np.inf
        
        # fix an element "k" and select the optimal solutions of the subproblems without k
        for k in rem_nodes:
            
            # Recursive call to the subproblems
            sub_cost,sub_node, duration =DnC(k,[x for x in rem_nodes if x!= k], timed, t1)
            
            # YOU CAN CHANGE THE MAXIMUM TIME HERE
            if timed:
                if time.time()-t1 > 600:
                    return sub_cost,sub_node, time.time() - t1            
            
            # Storing the optimal solutions in temporary variables
            if sub_cost < min_cost:
                index_opt = k
                opt_nodes = sub_node
                min_cost = sub_cost           
        
        # Update de cost
        cost = M[i -1,index_opt -1] + min_cost  
        
        # Return the optimal cost and the path that allowed us to find it
        return cost,  opt_nodes + [i], time.time() - t1