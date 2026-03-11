import torch
import numpy as np
from torch_geometric.data import Data

nodes = np.load("data/graph_data/nodes.npy")
edges = np.load("data/graph_data/edges.npy")

x = torch.tensor(nodes,dtype=torch.float)

edge_index = torch.tensor(edges.T,dtype=torch.long)

data = Data(x=x,edge_index=edge_index)

print(data)