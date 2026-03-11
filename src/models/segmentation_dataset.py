import os
import cv2
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(image_dir)

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        image_path = os.path.join(self.image_dir, file)
        mask_path = os.path.join(self.mask_dir, file.replace(".tif",".png"))

        with rasterio.open(image_path) as src:
            img = src.read()
            img = np.transpose(img,(1,2,0))

        img = img.astype(np.float32)/255.0

        mask = cv2.imread(mask_path,0)
        mask = mask/255.0

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask