import numpy as np
from scipy.spatial.distance import cdist

nodes = np.load("data/graph_data/nodes.npy")

distance_matrix = cdist(nodes,nodes)

edges = []

threshold = 30

for i in range(len(nodes)):

    for j in range(len(nodes)):

        if i != j and distance_matrix[i,j] < threshold:

            edges.append([i,j])

edges = np.array(edges)

np.save("data/graph_data/edges.npy",edges)

print("Graph edges:",len(edges))