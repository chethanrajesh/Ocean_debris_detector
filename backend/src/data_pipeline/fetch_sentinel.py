"""
fetch_sentinel.py
Authenticates to Google Earth Engine (GEE) using a service account,
retrieves Sentinel-2 MSI SR Harmonized imagery over ocean regions,
computes the Floating Debris Index (FDI) and Plastic Index (PI),
and saves per-node debris probability scores.

Band wavelength centres used in FDI formula (nm)
-------------------------------------------------
B4  = 664.9   (Red)
B6  = 740.2   (Red Edge 2) — note: S2-SR-Harmonized band numbering
B8A = 864.8   (NIR narrow / Vegetation Red Edge)
B11 = 1610.4  (SWIR-1)

Floating Debris Index (Biermann et al. 2020):
  FDI = B8A - (B6 + (B11 - B6) × ((832.9 - 664.9) / (1610.4 - 664.9)))

Plastic Index (Zietsman et al. 2022):
  PI = B4 / (B4 + B6)

Environment Variables Required
-------------------------------
GEE_SERVICE_ACCOUNT_KEY : path to GEE service account JSON key file
"""
import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# Band wavelength centres (nm) — Sentinel-2 MSI
LAMBDA_B4  = 664.9
LAMBDA_B6  = 740.2
LAMBDA_B8A = 864.8   # used as 832.9 in the FDI formula's linear interpolation anchor
LAMBDA_B11 = 1610.4

FDI_INTERP_FACTOR = (832.9 - LAMBDA_B4) / (LAMBDA_B11 - LAMBDA_B4)


def _init_gee() -> None:
    """Initialize GEE with service account credentials."""
    import ee
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if not key_path or not os.path.exists(key_path):
        raise EnvironmentError(
            "GEE_SERVICE_ACCOUNT_KEY must point to a valid service account JSON file."
        )
    with open(key_path) as f:
        info = json.load(f)
    credentials = ee.ServiceAccountCredentials(info["client_email"], key_path)
    ee.Initialize(credentials)
    logger.info(f"GEE initialized as {info['client_email']}")


def _compute_fdi_pi(image):
    """
    GEE server-side function: compute FDI and PI bands on a Sentinel-2 image.
    Adds bands 'FDI' and 'PI' and returns the image.
    """
    import ee

    # S2-SR-Harmonized band names: B4, B6, B8A, B11 (scaled 0–10000)
    B4  = image.select("B4").toFloat().divide(10000)
    B6  = image.select("B6").toFloat().divide(10000)
    B8A = image.select("B8A").toFloat().divide(10000)
    B11 = image.select("B11").toFloat().divide(10000)

    fdi = B8A.subtract(
        B6.add(B11.subtract(B6).multiply(FDI_INTERP_FACTOR))
    ).rename("FDI")

    pi = B4.divide(B4.add(B6).add(1e-6)).rename("PI")

    return image.addBands(fdi).addBands(pi)


def fetch_sentinel_debris_scores(
    days_back: int = 90,
    cloud_cover_max: float = 20.0
) -> np.ndarray:
    """
    Build a global Sentinel-2 composite, compute FDI and PI,
    and return per-node debris probability scores on a 0.25° grid.

    Debris probability = mean of scaled FDI (0–1) and PI (0–1).

    Returns
    -------
    scores : (N_lat, N_lon) float32 in [0, 1]
    """
    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api not installed. Run: pip install earthengine-api")

    _init_gee()

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    logger.info(f"Building Sentinel-2 composite {start_str} → {end_str}, cloud < {cloud_cover_max}%")

    # Load and filter collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_str, end_str)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        .filterBounds(ee.Geometry.Rectangle([-180, -60, 180, 60]))  # ocean extents
        .select(["B4", "B6", "B8A", "B11"])
        .map(_compute_fdi_pi)
    )

    count = collection.size().getInfo()
    logger.info(f"  Found {count} Sentinel-2 scenes.")

    if count == 0:
        logger.warning("No Sentinel-2 scenes found; returning zero scores.")
        n_lat = int(180 / RESOLUTION)
        n_lon = int(360 / RESOLUTION)
        return np.zeros((n_lat, n_lon), dtype=np.float32)

    # Median composite
    composite = collection.median()

    # Sample on a 0.25° grid
    n_lat = int(120 / RESOLUTION)   # -60 to +60
    n_lon = int(360 / RESOLUTION)

    sample_lats = np.linspace(-59.875, 59.875, n_lat, dtype=np.float32)
    sample_lons = np.linspace(-179.875, 179.875, n_lon, dtype=np.float32)

    scores = np.zeros((int(180 / RESOLUTION), n_lon), dtype=np.float32)
    lat_offset = int(30 / RESOLUTION)  # offset for -90...-60 rows

    # Build GEE point feature collection (sample at every 2° to avoid quota limits)
    step = int(2.0 / RESOLUTION)
    points = []
    lat_idxs = list(range(0, n_lat, step))
    lon_idxs = list(range(0, n_lon, step))

    for li in lat_idxs:
        for lj in lon_idxs:
            lat = float(sample_lats[li])
            lon = float(sample_lons[lj])
            points.append(
                ee.Feature(ee.Geometry.Point([lon, lat]),
                           {"li": li, "lj": lj})
            )

    fc = ee.FeatureCollection(points)
    sampled_fc = composite.select(["FDI", "PI"]).sampleRegions(
        collection=fc, scale=1000, geometries=False
    )

    try:
        info = sampled_fc.getInfo()
        features = info.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            li = int(props.get("li", 0))
            lj = int(props.get("lj", 0))
            fdi_val = props.get("FDI", 0.0) or 0.0
            pi_val  = props.get("PI", 0.0) or 0.0
            # Normalize: FDI typically -0.05 to +0.1 → scale to [0,1]
            fdi_norm = float(np.clip((fdi_val + 0.05) / 0.15, 0, 1))
            pi_norm  = float(np.clip(pi_val, 0, 1))
            prob = (fdi_norm + pi_norm) / 2.0
            scores[li + lat_offset, lj] = prob
    except Exception as exc:
        logger.error(f"GEE sampling failed: {exc}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "sentinel_scores.npy"
    np.save(str(out_path), scores)
    logger.info(f"Sentinel-2 debris scores saved to {out_path} — shape {scores.shape}")
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scores = fetch_sentinel_debris_scores(days_back=90)
    print(f"Sentinel scores shape: {scores.shape}, mean: {scores.mean():.4f}")
