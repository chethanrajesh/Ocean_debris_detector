import os
import cv2
import numpy as np

mask_dir = "data/final_segmentation_dataset/masks"

files = os.listdir(mask_dir)

nodes = []

for file in files:

    path = os.path.join(mask_dir,file)

    mask = cv2.imread(path,0)

    num_labels,labels = cv2.connectedComponents(mask)

    for label in range(1,num_labels):

        coords = np.column_stack(np.where(labels==label))

        center = coords.mean(axis=0)

        nodes.append(center)

nodes = np.array(nodes)

print("Total clusters:",len(nodes))

os.makedirs("data/graph_data",exist_ok=True)

np.save("data/graph_data/nodes.npy",nodes)