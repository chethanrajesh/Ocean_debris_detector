import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from ocean_dataset import OceanDataset

dataset_path = "data/processed/clean_satellite_dataset"

dataset = OceanDataset(dataset_path)

loader = DataLoader(dataset,batch_size=16,shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model("vit_small_patch16_224", pretrained=True)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 5

for epoch in range(epochs):

    print("Epoch:",epoch+1)

    for imgs in tqdm(loader):

        imgs = imgs.to(device)

        outputs = model(imgs)

        loss = outputs.mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

torch.save(model.state_dict(),"models/pretrained/pretrained_feature_encoder.pth")

print("Training complete")