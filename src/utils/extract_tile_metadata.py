import os
import json

# Get project root automatically
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

dataset_folder = os.path.join(project_root, "data", "processed", "clean_satellite_dataset")
output_path = os.path.join(project_root, "data", "processed", "tile_metadata.json")

print("Dataset folder:", dataset_folder)

if not os.path.exists(dataset_folder):
    print("ERROR: dataset folder not found")
    exit()

all_files = os.listdir(dataset_folder)
print("Total files found:", len(all_files))

metadata = {}

for file in all_files:
    if file.lower().endswith(".png") or file.lower().endswith(".tif"):

        metadata[file] = {
            "latitude": None,
            "longitude": None,
            "timestamp": "2024-06"
        }

print("Metadata entries created:", len(metadata))

with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved to:", output_path)