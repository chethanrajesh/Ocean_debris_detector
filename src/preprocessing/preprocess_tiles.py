import os
import rasterio
import numpy as np
import cv2
from tqdm import tqdm

input_folder = "data/raw/satellite_images"
output_folder = "data/processed/clean_satellite_dataset"

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

print("Raw tiles:", len(files))

clean_count = 0
removed_count = 0

for file in tqdm(files):

    path = os.path.join(input_folder, file)

    try:

        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1,2,0))

    except:
        removed_count += 1
        continue

    if img.shape[2] != 3:
        removed_count += 1
        continue

    # Remove mostly black tiles
    black_pixels = np.sum(np.all(img < 10, axis=2))
    black_ratio = black_pixels / (img.shape[0] * img.shape[1])

    if black_ratio > 0.5:
        removed_count += 1
        continue

    # Cloud detection
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    cloud_pixels = np.sum(gray > 220)
    cloud_ratio = cloud_pixels / gray.size

    if cloud_ratio > 0.5:
        removed_count += 1
        continue

    # Normalize pixel values
    img = img.astype(np.float32) / 255.0

    # Resize to standard size
    img = cv2.resize(img, (224,224))

    save_path = os.path.join(output_folder, file)

    img_uint8 = (img * 255).astype(np.uint8)

    cv2.imwrite(save_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    clean_count += 1


print("\nPreprocessing complete")
print("Clean images:", clean_count)
print("Removed images:", removed_count)