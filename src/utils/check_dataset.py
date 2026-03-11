import os
import random
import cv2
import matplotlib.pyplot as plt

dataset_path = "data/processed/clean_satellite_dataset"

files = [f for f in os.listdir(dataset_path) if f.endswith(".png")]

print("Total images:", len(files))

# pick 6 random images
sample_files = random.sample(files, 6)

plt.figure(figsize=(12,8))

for i, file in enumerate(sample_files):

    path = os.path.join(dataset_path, file)

    img = cv2.imread(path)

    # convert BGR → RGB for correct display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(file)
    plt.axis("off")

plt.tight_layout()
plt.show()