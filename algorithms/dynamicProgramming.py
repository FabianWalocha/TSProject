import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (DynamicProgramming).
def DynamicProgramming(graph, timed = False):
    
    adj_matrix = graph.weighted_adjacency_matrix
    
    t1 = time.time()
    
    M = np.matrix(adj_matrix)
    
    for i in range(0,np.size(M,1)):
        M[i,i] = np.inf       
    
    n = np.size(M,1)

    L = list(range(2,n+1))

    full_set = tuple([1,tuple(range(2,len(L)+2))])
    
    # Initialization
    cost_memo_child = {}
    path_memo_child = {}    
    cost_memo_parent = {}
    path_memo_parent = {}        
    
    # 1: LEAF NODES
    S_children = list(itertools.combinations(L,1))
    #We compute the edges from the leaves to the starting point
    for k in S_children[0:len(L)]:
        cost_memo_child[k] = M[k[0]-1,0]
        path_memo_child[k] = [k[0]]    
    
    # 2: LEAF PARENT NODES
    S_parent = []
    for j in L:
        exclude = [] + L
        exclude.remove(j)
        combos = list(itertools.combinations(exclude, 1))
        for k in combos:
            s = tuple([j,k])
            S_parent += [s]
            # Update memos
            cost_memo_parent[s] = M[j-1,k[0]-1] + cost_memo_child[k]
            path_memo_parent[s] = [j] + path_memo_child[k]        

#     t1 = time.time()
    
    # 3. GENERAL ALGORITHM

    for i in range(2,len(L)):
        
        cost_memo_child = cost_memo_parent
        path_memo_child = path_memo_parent
        cost_memo_parent = {}
        path_memo_parent = {}
        
#         print(cost_memo_child)
        
        for j in L:
            exclude = [] + L
            exclude.remove(j)
            combos = list(itertools.combinations(exclude, i))  
            
            for k in combos:
                
                # Initialize min cost
                min_cost = np.inf                   
                
                for l in k:

                    # We rewrite the available paths as an already solved problems
                    avail_nodes = list(k)
                    avail_nodes.remove(l)
                    s = tuple([l,tuple(avail_nodes)])   

                    # Compute the cost
                    cost = M[j-1,l-1] + cost_memo_child[s]

                    # YOU CAN CHANGE THE MAXIMUM TIME HERE
                    if timed:
                        if time.time()-t1 > 600:
                            return cost_memo_child[s], path_memo_child[s], time.time() - t1  

                    if cost < min_cost:
                        min_cost = cost
                        min_path = [j] + path_memo_child[s]                
                
                # This is the real tuple of the parent nodes subproblem
                z = tuple([j,k])
                    
                # Update memos
                cost_memo_parent[z] = min_cost
                path_memo_parent[z] = min_path

    # 4. FULL SET
    cost_memo_child = cost_memo_parent
    path_memo_child = path_memo_parent
    cost_memo_parent = {}
    path_memo_parent = {}                


    # Initialize min cost
    min_cost = np.inf                   

    k = full_set[1]
    
    for l in k:

        # We rewrite the available paths as an already solved problems
        avail_nodes = list(k)
        avail_nodes.remove(l)
        s = tuple([l,tuple(avail_nodes)])   

        # Compute the cost
        cost = M[0,l-1] + cost_memo_child[s]

        # YOU CAN CHANGE THE MAXIMUM TIME HERE
        if timed:
            if time.time()-t1 > 600:
                return cost_memo_child[s], path_memo_child[s], time.time() - t1  

        if cost < min_cost:
            min_cost = cost
            min_path = [1] + path_memo_child[s]                  
    
    path_memo_parent[full_set] = min_path + [1]
    cost_memo_parent[full_set] = min_cost
    
    # 5. RETURN
    
    return cost_memo_parent[full_set], path_memo_parent[full_set], time.time() - t1