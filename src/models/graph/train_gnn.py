import torch
from torch.optim import Adam
from tqdm import tqdm

from gat_model import GATModel
from load_graph import data


device = "cuda" if torch.cuda.is_available() else "cpu"

model = GATModel().to(device)

data = data.to(device)

optimizer = Adam(model.parameters(),lr=0.005)

epochs = 50

print("Using device:",device)

for epoch in range(epochs):

    optimizer.zero_grad()

    out = model(data.x,data.edge_index)

    loss = out.mean()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print("Epoch",epoch,"Loss",loss.item())


torch.save(model.state_dict(),"models/plastic_spread_gnn.pth")

print("GNN training complete")