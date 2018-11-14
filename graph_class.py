#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

class vertex:
    def __init__(self, vertex_name):
        self.neighbourhood = []
        # ANY kind of ID. For Heidelberg, could be latitude, longitude, altitude...
        self.name = vertex_name
    def is_neighbour_of(self, vertex):
        self.neighbourhood.append(vertex)
    def print(self):
        print(self.name)        
        
class edge:
    def __init__(self, a, b, w):
        self.v1 = a
        self.v2 = b
        self.weight = w      
    def print(self):
        print(self.v1.name, "to", self.v2.name, "taking", self.weight, "km")
    
class graph:  
    def __init__(self,v1,v2,w):
        self.vertices = []
        self.dic_vertices = {}
        self.edges = []
        self.dic_edges = {}
        #adjacency matrix is composed of np.infty and weights; 
        self.weighted_adjacency_matrix = np.ones((1))*np.infty 
        
        #symmetric TSP        
        v1.is_neighbour_of(v2)
        v2.is_neighbour_of(v1)
        
        self.dic_edges[(v1.name,v2.name)]=len(self.edges)
        self.edges.append(edge(v1,v2,w))
                
        self.dic_vertices[v1.name]=len(self.vertices)
        self.vertices.append(v1)
        self.dic_vertices[v2.name]=len(self.vertices)
        self.vertices.append(v2)   
                
        a = (len(self.vertices), len(self.vertices))
        
        temp_matrix = np.ones(a)*np.infty
        #symmetric TSP                      
        temp_matrix[self.dic_vertices[v1.name], self.dic_vertices[v2.name]] = w
        temp_matrix[self.dic_vertices[v2.name], self.dic_vertices[v1.name]] = w

        temp_matrix[:-1,:-1] = self.weighted_adjacency_matrix
        self.weighted_adjacency_matrix =np.copy(temp_matrix)
        
        
    def attach_vertex(self, new_vertex, existing_vertex, weight):        

        new_edge = edge(new_vertex, existing_vertex, weight)
        
        self.dic_edges[(new_edge.v1.name,new_edge.v2.name)] = len(self.edges) 
        self.edges.append(new_edge)
        
        self.dic_vertices[new_vertex.name]=len(self.vertices)
        self.vertices.append(new_vertex)
        
        
        #append a column and a row to the symmetric adjacency matrix, where the (new_vertex, existing_vertex) entries
        #are updated with weight, everything else set to zero
        
        #dictionary of names to indices makes the substitution straightforward
             
        # new matrix with an extra row and column
        a = (len(self.vertices), len(self.vertices))
        temp_matrix = np.ones(a)*np.infty
        # copy the weight to the appropriate place and vice versa (symmetric TSP)
        temp_matrix[self.dic_vertices[new_vertex.name],self.dic_vertices[existing_vertex.name]] = weight
        temp_matrix[self.dic_vertices[existing_vertex.name],self.dic_vertices[new_vertex.name]] = weight
        #and copy the old matrix entries onto the new matriz entries
        temp_matrix[:-1,:-1] = self.weighted_adjacency_matrix
        self.weighted_adjacency_matrix =np.copy(temp_matrix)
        
        #symmetric TSP
        self.vertices[self.dic_vertices[new_vertex.name]].is_neighbour_of(self.vertices[self.dic_vertices[existing_vertex.name]])
        self.vertices[self.dic_vertices[existing_vertex.name]].is_neighbour_of(self.vertices[self.dic_vertices[new_vertex.name]])
             
    def attach_edge(self, v1, v2, weight):
        
        new_edge = edge(v1, v2, weight)
        self.dic_edges[(new_edge.v1.name,new_edge.v2.name)] = len(self.edges) 

        self.edges.append(new_edge)
        #symmetric TSP
        self.weighted_adjacency_matrix[self.dic_vertices[v1.name],self.dic_vertices[v2.name]] = weight
        self.weighted_adjacency_matrix[self.dic_vertices[v2.name],self.dic_vertices[v1.name]] = weight

        self.vertices[self.dic_vertices[v1.name]].is_neighbour_of(self.vertices[self.dic_vertices[v2.name]])
        self.vertices[self.dic_vertices[v2.name]].is_neighbour_of(self.vertices[self.dic_vertices[v1.name]])
        
    def remove_edge(self, v1, v2):
        #symmetric TSP
        # remove edge and dictionary reference from the graph
        try:
            index = self.dic_edges[(v1.name,v2.name)]
        except:
            index = self.dic_edges[(v2.name,v1.name)]
        del self.edges[index]
        
        try:
            del self.dic_edges[(v1.name,v2.name)]
        except:
            del self.dic_edges[(v2.name,v1.name)]
            
        # update adjacency matrix entries to infinity
        self.weighted_adjacency_matrix[self.dic_vertices[v1.name],self.dic_vertices[v2.name]] = np.infty
        self.weighted_adjacency_matrix[self.dic_vertices[v2.name],self.dic_vertices[v1.name]] = np.infty
        # remove vertices from each other's neighbourhood
        self.vertices[self.dic_vertices[v1.name]].neighbourhood.remove(self.vertices[self.dic_vertices[v2.name]])
        self.vertices[self.dic_vertices[v2.name]].neighbourhood.remove(self.vertices[self.dic_vertices[v1.name]])

        
    def update_edge(self, v1, v2, new_weight):
        #symmetric TSP
        try:
            index = self.dic_edges[(v1.name,v2.name)]
        except:
            index = self.dic_edges[(v2.name,v1.name)]
            
        self.edges[index].weight = new_weight   
        self.weighted_adjacency_matrix[self.dic_vertices[v1.name],self.dic_vertices[v2.name]] = new_weight
        self.weighted_adjacency_matrix[self.dic_vertices[v2.name],self.dic_vertices[v1.name]] = new_weight
        
    def get_edge(self,v1,v2):
        """returns a list with the index and weight of the edge between v1 and v2 (symmetric)"""
        try:
            index = self.dic_edges[(v1.name,v2.name)]
        except:
            index = self.dic_edges[(v2.name,v1.name)]
        return [index, self.edges[index].weight]
                    
    def attach_vertex_fully_connected(self, new_vertex, weight_function):
        """used to build fully connected graphs (Heidelberg)"""
        vertex_list = self.vertices.copy()
        self.attach_vertex(new_vertex, vertex_list[0], weight_function(new_vertex, vertex_list[0]))
        
        for existing_vertex in vertex_list[1:]:
            self.attach_edge(new_vertex, existing_vertex, weight_function(new_vertex, existing_vertex))
            
    def get_path_weight(self, path):
        if type(path[0])!= vertex:
            path = [vertex(name) for name in path]            
        path_weight = 0
        previous_node = path[0]
        for node in path[1:]:
            path_weight += self.get_edge(node,previous_node)[1]
            previous_node = node
        return path_weight
    def get_cycle_weight(self, path):
        """joins the first and the last vertices when calculating weight"""
        return self.get_path_weight(path) + self.get_path_weight((path[-1],path[0]))
            
    def print(self):
        """prints general info"""
        names = [a.name for a in self.vertices]
        print(len(self.vertices), "VERTICES:",', '.join(names),"\n")
        weights = [str(a.weight)+" km" for a in self.edges]
        print(len(self.edges), "EDGES:",', '.join(weights))
        print("\nADJACENCY MATRIX:\n",self.weighted_adjacency_matrix)


def get_node_coordinates_2D(filename):
    """gets node coordinates in 2D. Should work with any distance"""
    node_coordinates = []

    with open(filename) as input_data:
        interesting_lines = []
    
        for line in input_data:
            if line.strip() == 'NODE_COORD_SECTION':  #read from this point
                break
            
        for line in input_data:
            if line.strip() == 'EOF': #until this point
                break
            else:
                interesting_lines.append(list(map(float,line.split()))) #some files have ints in floating point notation

    input_data.close()
    node_coordinates = np.array(interesting_lines)[:,[1,2]]
    node_coordinates = [(int(a) for a in b) for b in node_coordinates]
    return node_coordinates


def nearest_int(x):
    #deprecated: using np.around instead
    if x-int(x)<int(x+1)-x:
        return int(x)
    else:
        return int(x+1)

def nearest_int_euclidean_distance_2D(v1,v2):
    """between vertices, rounded to the nearest integer,
    as required in the TSPLIB docs"""
    xd = v1.name[0] - v2.name[0]
    yd = v1.name[1] - v2.name[1]
    return int(np.around(np.sqrt(xd*xd + yd*yd)))


def heidelberg_2D(filename):
    """parses 2D distance TSP type .tsp file from Heidelberg and returns the graph"""
    nodes = get_node_coordinates_2D(filename)
    v1 = vertex(tuple(nodes[0]))
    v2 = vertex(tuple(nodes[1]))
    heidelberg_graph = graph(v1,v2, nearest_int_euclidean_distance_2D(v1,v2))
    for node in nodes[2:]:
        heidelberg_graph.attach_vertex_fully_connected(vertex(tuple(node)), nearest_int_euclidean_distance_2D)
    return heidelberg_graph

def fully_connected_graph_from_coordinate_list(nodes, distance = nearest_int_euclidean_distance_2D):
    """creates a fully connected graph from a coordinate list; defaults to nearest_int_euclidean_distance"""
    v1 = vertex(tuple(nodes[0]))
    v2 = vertex(tuple(nodes[1]))
    output_graph = graph(v1,v2, distance(v1,v2))
    for node in nodes[2:]:
        output_graph.attach_vertex_fully_connected(vertex(tuple(node)), distance)
    return output_graph

def graph_from_adjacency_matrix(matrix):
    """creates a fully connected graph from an adjacency matrix in the form of a numpy array"""
    nodes = [i for i in range(matrix.shape[0])]
    v1 = vertex(nodes[0])
    v2 = vertex(nodes[1])
    output_graph = graph(v1,v2, matrix[0][1])
    for node in nodes[2:]:
        output_graph.attach_vertex_fully_connected(vertex(node), lambda x,y: np.infty)
    for i in nodes:
        for j in nodes:
            if i<j:
                output_graph.update_edge(vertex(i),vertex(j),matrix[i][j])
    return output_graph



def heidelberg_optimal_tour(filename):
    """gets node indices for optimal path from opt.tour file"""
    node_indices = []

    with open(filename) as input_data:
        interesting_lines = []
    
        for line in input_data:
            if line.strip() == 'TOUR_SECTION':  #read from this point
                break
            
        for line in input_data:
            if line.strip() == '-1': #until this point
                break
            else:
                interesting_lines.append(int(line.split()[0]))

    input_data.close()
    node_indices = [index-1 for index in interesting_lines]
    return node_indices

def plotTSP_2D(path):
    """
    path: ordered list of vertices
    """       
    x = []; y = []
    for node in path:
        x.append(node.name[0])
        y.append(node.name[1])

    # Set a scale for the arrow heads
    a_scale = float(min(max(x)-min(x), max(y)-min(y)))/float(50)
    
    #plot the vertices
    plt.plot(x, y, linestyle = 'none', color = 'black', marker = '.', fillstyle='none', alpha=0.8)

    #Draw the path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
           color ='r', length_includes_head=True, ls='--', aa=True, alpha =0.5)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
               color = 'r', length_includes_head = True, ls='--', aa=True, alpha=0.5)
    plt.show()
    
def brute_force(graph, max_iterations = np.infty, random_init = True, return_graph = True):
    
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
        node_output = plotTSP_2D(min_path)
    else:
        node_output = min_path
    return end-start, min_weight, node_output


# simulated annealing implementation
def simulated_annealing(graph,random_init = True, temperature = 100000, return_graph = True):
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
        if np.exp(-(test_min-current_min)**2/(temperature-i))> np.random.rand():
            current_min = test_min
        else:
            current_min = min(current_min, test_min)    
    
    end = time.time()
    if return_graph:
        node_output = plotTSP_2D(nodes)
    else:
        node_output = nodes
    return end-start, current_min, node_output



