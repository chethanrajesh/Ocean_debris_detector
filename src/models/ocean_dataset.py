import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class OceanDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith(".tif")]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = os.path.join(self.folder, self.files[idx])

        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img,(1,2,0)) 

        img = img.astype(np.uint8)

        img = self.transform(img)

        return img