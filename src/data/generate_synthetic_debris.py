import os
import cv2
import numpy as np
import rasterio
from tqdm import tqdm

input_folder = "data/processed/clean_satellite_dataset"

output_image_folder = "data/segmentation_dataset/images"
output_mask_folder = "data/segmentation_dataset/masks"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

print("Input folder:", input_folder)
print("Tiles found:", len(files))
print("Tiles available:", len(files))


def generate_blob_mask(size=224):

    mask = np.zeros((size,size), dtype=np.uint8)

    num_blobs = np.random.randint(1,5)

    for _ in range(num_blobs):

        center = (
            np.random.randint(0,size),
            np.random.randint(0,size)
        )

        radius = np.random.randint(5,20)

        cv2.circle(mask, center, radius, 1, -1)

    return mask


for file in tqdm(files):

    path = os.path.join(input_folder, file)

    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img,(1,2,0))

    img = img.astype(np.uint8)

    mask = generate_blob_mask()

    synthetic = img.copy()

    debris_pixels = mask == 1

    synthetic[debris_pixels] = synthetic[debris_pixels] + np.random.randint(30,60)

    synthetic = np.clip(synthetic,0,255)

    image_out = os.path.join(output_image_folder, file)

    mask_out = os.path.join(output_mask_folder, file.replace(".tif",".png"))

    cv2.imwrite(image_out, cv2.cvtColor(synthetic, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_out, mask*255)

print("Synthetic dataset generation complete")