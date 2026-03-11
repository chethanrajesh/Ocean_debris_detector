import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class TemporalDataset(Dataset):

    def __init__(self, mask_dir):

        self.mask_dir = mask_dir
        self.files = os.listdir(mask_dir)

    def __len__(self):

        return len(self.files)-2

    def __getitem__(self, idx):

        seq = []

        for i in range(3):

            path = os.path.join(self.mask_dir,self.files[idx+i])

            mask = cv2.imread(path,0)

            mask = mask/255.0

            seq.append(mask)

        seq = np.array(seq)

        seq = torch.tensor(seq).unsqueeze(1).float()

        target = seq[-1]

        return seq,target