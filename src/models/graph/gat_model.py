import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(2,16,heads=4)

        self.gat2 = GATConv(64,16)

        self.out = nn.Linear(16,1)


    def forward(self,x,edge_index):

        x = self.gat1(x,edge_index)

        x = torch.relu(x)

        x = self.gat2(x,edge_index)

        x = torch.relu(x)

        x = self.out(x)

        return x