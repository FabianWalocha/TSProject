import time
import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline

filename = input("Write the name of data like this type( eil51_json_array.txt ):  ")
with open(filename, 'r') as myfile:
    data=myfile.read().replace('\n', '')
data = eval(data)
data = np.array(data)

iteration = int(input("Please enter the number of iterations to execute random approach:  "))
path_distance = lambda r, c: np.sum([int(np.linalg.norm(c[r[p]] - c[r[p - 1]])+1) for p in range(len(r))])
route = np.arange(data.shape[0])
best_distance = path_distance(route, data)
last_route = list(route)
first_route = list(route)

start_time = time.time()
for i in range(iteration):
    np.random.shuffle(route)
    test_route = route
    new_distance = path_distance(test_route, data)
    if new_distance < best_distance:
        last_route = list(test_route)
        best_distance = new_distance
finish_time = time.time()
duration = finish_time-start_time
last_route.append(last_route[0])




print("The route is :", last_route)
print("Number of iteration is: ",iteration)
print("Total execution time is: ",duration," seconds")
print("Total distance is:", best_distance)
