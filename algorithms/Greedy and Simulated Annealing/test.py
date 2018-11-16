import time
from anneal import SimAnneal
import matplotlib.pyplot as plt
import random

coords = []
with open('att48.txt','r') as f:
    i = 0
    for line in f.readlines():
        line = [float(x.replace('\n','')) for x in line.split(' ')]
        coords.append([])
        for j in range(1,3):
            coords[i].append(line[j])
        i += 1
start_time = time.time()
if __name__ == '__main__':
    sa = SimAnneal(coords, stopping_iter = 4000)
    sa.anneal()
    sa.visualize_routes()
    sa.plot_learning()
    
end_time = time.time()
duration = end_time - start_time
print("Execution time is :",duration)