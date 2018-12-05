import numpy as np
import time
import itertools

# To run the classical Branch and Bound method, use this function (BranchAndBoundBinaryTree).
def BranchAndBoundBinaryTree(graph, timed = False):

    adj_mat = graph.weighted_adjacency_matrix
    adj_mat = np.matrix(adj_mat)
    
    for i in range(0,np.size(adj_mat,1)):
        adj_mat[i,i] = np.inf    
    
    t1 = time.time()
    cost, path, duration = BnBBinaryTree(adj_mat, timed, t1)
        
    return cost, path, duration 

# The algorithm
def BnBBinaryTree(M,timed, t1):
    
    # Matrix reduction step
    f,MR = mat_redux(M,0)
    
    n = np.size(M,1)

    cost, edges_col, opt_edge_col, duration = BnBBinaryTreeRecursive(MR,[],f,np.inf, [], n, timed, t1)
    
    return cost, create_path_from_edges(opt_edge_col)+[1], duration  

# This is the recursive algorithm that is called after pre-processing the matrix    
def BnBBinaryTreeRecursive(M, edges_collection, cost, opt_cost, opt_edge_col, n, timed, t1):

    # YOU CAN CHANGE THE MAXIMUM TIME HERE
    if timed:
        if time.time()-t1 > 600:
            return opt_cost, edges_collection, opt_edge_col, time.time() - t1        
    # If the optimum cost is less than the actual cost we are not supposed to do anything
    if opt_cost <= cost:
        return opt_cost, edges_collection, opt_edge_col, time.time() - t1  
    
    # We know that we cannot find a solution if there are less nodes than cities to visit
    if n-len(edges_collection) > np.sum(M.min(axis=1) == 0):
        return opt_cost, edges_collection, opt_edge_col, time.time() - t1  
    
    # If the search is complete, a solution has been found
    elif n-len(edges_collection) == np.sum(M.min(axis=1) == 0) == 0:
        if cost < opt_cost:
            if len(create_path_from_edges(edges_collection)) != 0:
                return cost, edges_collection, edges_collection, time.time() - t1  
            else:
                return opt_cost, edges_collection, opt_edge_col, time.time() - t1  
        else:
            return opt_cost, edges_collection, opt_edge_col, time.time() - t1  
    
    # Edge selection
    edge = select_edges(M, n)
    
    if edge == (0,0):
        return opt_cost, edges_collection, opt_edge_col, time.time() - t1  
    
    # Children trees creation
    f_L, L, f_R, R = children_subtrees(M, edge, cost, )
    
    # We search the trees
    opt_cost, edges_col, opt_edge_col, duration = BnBBinaryTreeRecursive(L, edges_collection + [edge], f_L, opt_cost, opt_edge_col, n, timed, t1)
    if opt_cost > f_R:
        opt_cost, edges_col, opt_edge_col, duration = BnBBinaryTreeRecursive(R, edges_collection, f_R, opt_cost, opt_edge_col, n, timed, t1)
        
    return opt_cost, edges_col, opt_edge_col, time.time() - t1  


# Helper functions used in the following algorithm

# This function calculates the normaliced matrix with at least one zero in each row and column.
# It also computes the cost of reduction.
def mat_redux(M, cost):
    
    # Used to reshape the minimum value arrays into matrices
    n = np.size(M,1)    
    
    # Reduction by rows
    m1 = M.min(axis=1)
    # to avoid inf-inf math error
    m1[m1 == np.inf] = 0
    Mrow = M - m1.reshape(n,1)
    
    # Reduction by columns
    m2 = Mrow.min(axis=0)
    m2[m2 == np.inf] = 0
    Mcol = Mrow - np.matrix(m2).reshape(1,n)
    
    # Lower bound calculation
    f = m1.sum() + m2.sum()
    
    return cost + f, Mcol

# This function selects the zero corresponding to the edge with the greatest weight
def select_edges(A, n):

    # We wish to know where are the pivots
    zeros_irow, zeros_icol = np.where(A==0)

    # Variables initialization
    max_edge_cost = 0
    opt_edge = (0,0)

    #  We traverse every zero such that it has the highest cost
    for k in range(len(zeros_irow)):
        
        i = zeros_irow[k]
        j = zeros_icol[k]
        x = [x for x in range(0,n) if x != i]
        y = [x for x in range(0,n) if x != j]

        edge_cost = np.min(A[i,y]) + np.min(A[x,j])   
        
        if edge_cost > max_edge_cost:
            
            max_edge_cost = edge_cost
            opt_edge = (i,j)

    return opt_edge

# This function created the children Left and Right subtree (with and without the edge)
def children_subtrees(M, edge, f):
    
    # Left and Right matrices
    L = np.copy(M)
    R = np.copy(M)

    # For the left child we add infinity to the column and row corresponding to the chosen edge
    try:
        L[edge[0],:] = np.inf
        L[:,edge[1]] = np.inf
        L[edge[1],edge[0]] = np.inf
    except:
        print(edge)
    # After this the matrix should be reduced
    f_L, L = mat_redux(L,f)

    # For the right child we just add infinity to the location of the node, so that it's excluded
    R[edge[0], edge[1]] = np.inf
    f_R, R = mat_redux(R,f)

    return f_L, L, f_R, R

# Function used to reconstruct the path from the given edges 
# (the edge list is assumed to be same size of the number of cities)
def create_path_from_edges(EdgeList):
    
    E = EdgeList + []
    
    n = len(E)
    path = [1]
    possible = True

    while True:

        continue_chain = 0

        for i in E:
            if i[0] == path[-1]-1:
                path.append(i[1]+1)
                continue_chain = 1
                break

        E.remove(i)

        if continue_chain == 0:
            return []
        if len(path) == n:
            return path