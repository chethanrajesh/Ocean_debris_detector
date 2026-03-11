import ee
import geemap
import os
# Initialize Earth Engine
ee.Initialize(project='practical-bebop-477117-a5')

# Output folder
output_dir = "data/raw/satellite_images"
os.makedirs(output_dir, exist_ok=True)

# Base ocean region (Arabian Sea example)
base_region = ee.Geometry.Rectangle([71.5, 17.5, 73.5, 19.5])

# Load updated Sentinel-2 dataset
dataset = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(base_region)
    .filterDate("2024-06-01", "2024-06-30")
    .sort("CLOUDY_PIXEL_PERCENTAGE")
)

# Tile size in degrees
tile_size = 0.1

# Bounding box coordinates
xmin, ymin, xmax, ymax = 71.5, 17.5, 73.5, 19.5

tile_id = 0

lat = ymin
while lat < ymax:
    lon = xmin
    while lon < xmax:

        region = ee.Geometry.Rectangle([lon, lat, lon + tile_size, lat + tile_size])

        # Select best image for this tile
        tile_dataset = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(region)
            .filterDate("2024-06-01", "2024-06-30")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        count = tile_dataset.size().getInfo()

        if count == 0:
            print(f"No image for tile {tile_id}")
            lon += tile_size
            tile_id += 1
            continue

        image = tile_dataset.first()

        image = image.select(["B4", "B3", "B2"]).divide(10000).multiply(255).toUint8()

        filename = f"{output_dir}/tile_{tile_id}.tif"

        try:
            geemap.ee_export_image(
                image,
                filename=filename,
                scale=10,
                region=region,
                file_per_band=False
            )

            print(f"Downloaded tile {tile_id}")

        except Exception as e:
            print(f"Failed tile {tile_id}: {e}")

        tile_id += 1
        lon += tile_size

    lat += tile_size