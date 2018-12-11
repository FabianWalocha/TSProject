import networkx as nx
#import matplotlib.pyplot as plt
import time
import sys,os
sys.path.insert(0,'../..')
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
#finding DSF respect to minimum spanning tree
from collections import defaultdict

global data_json


filename = input("Write file name as this type (../../data/eil51_json_array.txt): ")

with open(filename, 'r') as myfile:
    data_json = myfile.read().replace('\n', '')
data_json = eval(data_json)
data_json = np.array(data_json)

global distance_data
distance_data = distance_matrix(data_json, data_json)
distance_data = distance_data.tolist()

t1 = time.time()

# Finding minimum spanning tree

class Graph():
    global parent_list
    global root_list
    global weight_list

    parent_list = []
    root_list = []
    weight_list = []

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def printMST(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            # print (parent[i],"-",i,"\t",self.graph[i][ parent[i] ] )
            parent_list.append(parent[i])
            root_list.append(i)
            weight_list.append(self.graph[i][parent[i]])

    def minKey(self, key, mstSet):
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def primMST(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mstSet = [False] * self.V
        parent[0] = -1
        for cout in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.printMST(parent)

g = Graph(len(distance_data))
g.graph = distance_data
g.primMST()


# converting df file
dictionary = {'Parent':parent_list,
              'Root':root_list,
              'weight':weight_list}
df = pd.DataFrame(dictionary)





class Graph:
    global opt_route
    opt_route = []

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUtil(self, v, visited):
        visited[v] = True
        opt_route.append(v)
        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i, visited)

    def DFS(self, v):

        visited = [False] * (len(self.graph))

        self.DFSUtil(v, visited)


g = Graph()
for number in range(len(df)):
    g.addEdge(df['Parent'][number], df['Root'][number])
for number in range(len(df)):
    g.addEdge(df['Root'][number], df['Parent'][number])

print("Following is DFS from (starting from vertex 5)")
g.DFS(5)

global total_distance
total_distance = 0
for i in range(len(opt_route)):
    try:
        city_1 = opt_route[i]
        city_2 = opt_route[i+1]
        total_distance = total_distance + int(distance_data[city_1][city_2])+1
    except:
        city_1 = opt_route[i]
        city_2 = opt_route[0]
        total_distance = total_distance + int(distance_data[city_1][city_2])+1

t2 = time.time()

global duration
duration = t2-t1





#print("The route is :", opt_route)
print("Total execution time is: ",duration," seconds")
print("Total distance is : ",total_distance)