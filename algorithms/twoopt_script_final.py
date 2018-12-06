import time
import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline

path_distance = lambda r, c: np.sum([int(np.linalg.norm(c[r[p]] - c[r[p - 1]])+1) for p in range(len(r))])
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

#duration = float(input("What's the limit of execution time?:  "))

def two_opt(cities,improvement_threshold=0.001): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    duration = 600
    route = np.arange(cities.shape[0])
    improvement_factor = 1
    time1 = time.time()
    best_distance = path_distance(route,cities)
    while improvement_factor > improvement_threshold:
        distance_to_beat = best_distance
        for swap_first in range(1,len(route)-2):
            for swap_last in range(swap_first+1,len(route)):
                time2 = time.time()
                if time2 - time1 > duration:
                    break
                new_route = two_opt_swap(route,swap_first,swap_last)
                new_distance = path_distance(new_route,cities)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                time2 = time.time()
                if time2 - time1 > duration:
                    break
        improvement_factor = 1 - best_distance/distance_to_beat
    time3 = time.time()
    return best_distance, route, (time3-time1)

# filename = input("Write the name of data.(i.e: eil51_json_array.txt):  ")
# with open(filename, 'r') as myfile:
#     data=myfile.read().replace('\n', '')
# data = eval(data)
# data = np.array(data)

# start_time = time.time()
# route = two_opt(data,0.001)
# end_time = time.time()
# duration2 = end_time - start_time
# route = list(route)
# route.append(route[0])
# best_distance = path_distance(route,data)






# print("The Route is : " ,route)
# print("Total execution time is: ",duration2," seconds")
# print("Total distance is : ",best_distance)