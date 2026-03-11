import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from convlstm_model import ConvLSTM
from temporal_dataset import TemporalDataset


mask_dir = "data/final_segmentation_dataset/masks"

dataset = TemporalDataset(mask_dir)

loader = DataLoader(dataset,batch_size=8,shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:",device)

model = ConvLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

loss_fn = torch.nn.BCEWithLogitsLoss()

epochs = 5

for epoch in range(epochs):

    print("Epoch:",epoch+1)

    for seq,target in tqdm(loader):

        seq = seq.to(device)
        target = target.to(device)

        pred = model(seq)

        loss = loss_fn(pred,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(),"models/debris_drift_model.pth")

print("ConvLSTM drift model trained")