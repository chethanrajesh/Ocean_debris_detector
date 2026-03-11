import os
import cv2
import numpy as np

mask_dir = "data/final_segmentation_dataset/masks"

files = os.listdir(mask_dir)

clusters = []

for file in files:

    path = os.path.join(mask_dir,file)

    mask = cv2.imread(path,0)

    _,labels = cv2.connectedComponents(mask)

    for label in range(1,labels.max()+1):

        coords = np.column_stack(np.where(labels==label))

        center = coords.mean(axis=0)

        clusters.append(center)

print("Total clusters detected:",len(clusters))