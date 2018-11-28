import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (BranchAndBound).
def BranchAndBound(graph, timed = False):

    adj_mat = graph.weighted_adjacency_matrix
    adj_mat = np.matrix(adj_mat)
    
    n = np.size(adj_mat,1)

    for i in range(0,n):
        adj_mat[i,i] = np.inf
    
    rem_nodes = list(range(2,n+1))
    cost = 0
    optimal_cost = np.inf
    v = [1] 
    
    t1 = time.time()
    cost, path, opt_cost, opt_path, duration = BnBClassic(adj_mat,v,rem_nodes,cost,optimal_cost, [], timed, t1)
        
    return opt_cost, opt_path, duration
    
        
def BnBClassic(A,v,rem_nodes, cost, optimal_cost, opt_path, timed, t1):
    
    # YOU CAN CHANGE THE MAXIMUM TIME HERE
    if timed:
        if time.time()-t1 > 600:
            return cost, v, optimal_cost, opt_path, time.time() - t1
    
    # End of the path
    if len(rem_nodes)==1:
        cost += A[v[-1]-1, rem_nodes[0]-1] + A[rem_nodes[0]-1,0]
        v += rem_nodes + [1]

        if cost < optimal_cost:
            return cost, v, cost, v, time.time() - t1
        else:
            return cost, v, optimal_cost, opt_path, time.time() - t1
    
    else:
        
        k = len(v)
        n = np.size(A,0)+1

        cost_branches = []
        path_branches = []
        f = []
        
        # Calculating the costs, the f and the paths
        for i in rem_nodes:
            cost_branches.append(cost + A[v[-1]-1,i-1])
            path_branches.append(v+[i])
            # In g we select the minimum distance we can be able to travel
            g = np.min(A[:,[x-1 for x in rem_nodes + [1] if x!= i]])
            if g == np.inf:
                continue
            f.append(cost_branches[-1] + (n-k)*
                     g)
        
        if len(f) == 0:
            return cost, v, optimal_cost, opt_path, time.time() - t1
        
        # Sorting the arrays
        order = np.argsort(f)
        cost_branches = [cost_branches[i] for i in order]
        path_branches = [path_branches[i] for i in order]
        f = [f[i] for i in order] 


        # We explore recursively the branches and check if an optimal solution can be found
        for i in range(len(f)):
            
            # We discard all of the branches that cannot decrease the cost function
            # As all of the branches are sorted by cost, the following branches after
            # a discarded one will also be discarded
            if f[i] >= optimal_cost:
                break
            else:
                rem_nodes_sub = [x for x in rem_nodes if x not in path_branches[i]]
                cost, v, optimal_cost, opt_path, duration = BnBClassic(A,path_branches[i],rem_nodes_sub, cost_branches[i], optimal_cost, opt_path, timed, t1)
            
        return cost, v, optimal_cost, opt_path, time.time()-t1