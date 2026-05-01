"""
fetch_sentinel.py
Computes Floating Debris Index (FDI) and Plastic Index (PI) from
Sentinel-2 MSI SR Harmonized imagery via Google Earth Engine.

FDI (Biermann et al. 2020):
  FDI = B8A - (B6 + (B11 - B6) × ((832.9 - 664.9) / (1610.4 - 664.9)))

PI (Zietsman et al. 2022):
  PI = B4 / (B4 + B6)

Output
------
data/sentinel_scores.npy : (720, 1440) float32  debris probability [0, 1]
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

FDI_INTERP = (832.9 - 664.9) / (1610.4 - 664.9)


def _init_gee():
    import ee
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "")
    if not key_path or not os.path.exists(key_path):
        raise EnvironmentError(
            f"GEE_SERVICE_ACCOUNT_KEY not found at: {key_path}"
        )
    with open(key_path) as f:
        info = json.load(f)
    creds = ee.ServiceAccountCredentials(info["client_email"], key_path)
    ee.Initialize(creds)
    logger.info(f"GEE initialized as {info['client_email']}")
    return ee


def fetch_sentinel_debris_scores(
    days_back: int = 90,
    cloud_cover_max: float = 20.0,
    sample_deg: float = 2.0,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Build a global Sentinel-2 FDI/PI composite and return per-cell
    debris probability on a 0.25° grid.

    Parameters
    ----------
    days_back       : composite window in days (default 90)
    cloud_cover_max : max cloud cover % per scene (default 20)
    sample_deg      : sampling resolution in degrees (default 2.0 — keeps
                      GEE quota usage low; interpolated to 0.25° after)
    force_refresh   : re-download even if cached file exists

    Returns
    -------
    scores : (720, 1440) float32 in [0, 1]
    """
    out_path = DATA_DIR / "sentinel_scores.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached Sentinel scores from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ee = _init_gee()

    now      = datetime.now(timezone.utc)
    end_str  = now.strftime("%Y-%m-%d")
    start_str = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    logger.info(f"Building S2 composite {start_str} → {end_str}, cloud<{cloud_cover_max}%")

    def _add_indices(image):
        B4  = image.select("B4").toFloat().divide(10000)
        B6  = image.select("B6").toFloat().divide(10000)
        B8A = image.select("B8A").toFloat().divide(10000)
        B11 = image.select("B11").toFloat().divide(10000)
        fdi = B8A.subtract(
            B6.add(B11.subtract(B6).multiply(FDI_INTERP))
        ).rename("FDI")
        pi = B4.divide(B4.add(B6).add(1e-6)).rename("PI")
        return image.addBands(fdi).addBands(pi)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_str, end_str)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        .filterBounds(ee.Geometry.Rectangle([-180, -60, 180, 60]))
        .select(["B4", "B6", "B8A", "B11"])
        .map(_add_indices)
    )

    composite = collection.median()

    # Sample on a coarse grid to stay within GEE quota
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    scores = np.zeros((n_lat, n_lon), dtype=np.float32)

    step = max(1, int(sample_deg / RESOLUTION))
    sample_lats = np.linspace(-59.875, 59.875, int(120 / RESOLUTION))[::step]
    sample_lons = np.linspace(-179.875, 179.875, n_lon)[::step]

    # Build feature collection of sample points
    points = []
    lat_indices = []
    lon_indices = []

    all_lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat)
    all_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon)

    for lat in sample_lats:
        li = int(np.argmin(np.abs(all_lats - lat)))
        for lon in sample_lons:
            lj = int(np.argmin(np.abs(all_lons - lon)))
            points.append(
                ee.Feature(
                    ee.Geometry.Point([float(lon), float(lat)]),
                    {"li": li, "lj": lj}
                )
            )
            lat_indices.append(li)
            lon_indices.append(lj)

    logger.info(f"Sampling {len(points)} points from GEE composite...")

    # Process in batches of 500 to avoid GEE memory limits
    BATCH = 500
    for b_start in range(0, len(points), BATCH):
        b_end = min(b_start + BATCH, len(points))
        fc = ee.FeatureCollection(points[b_start:b_end])
        try:
            sampled = composite.select(["FDI", "PI"]).sampleRegions(
                collection=fc, scale=1000, geometries=False
            )
            info = sampled.getInfo()
            for feat in info.get("features", []):
                props = feat.get("properties", {})
                li = int(props.get("li", 0))
                lj = int(props.get("lj", 0))
                fdi_val = float(props.get("FDI") or 0.0)
                pi_val  = float(props.get("PI")  or 0.0)
                # Normalise FDI: typical range -0.05 to +0.10
                fdi_norm = float(np.clip((fdi_val + 0.05) / 0.15, 0, 1))
                pi_norm  = float(np.clip(pi_val, 0, 1))
                scores[li, lj] = (fdi_norm + pi_norm) / 2.0
            logger.info(f"  Batch {b_start}–{b_end} OK")
        except Exception as exc:
            logger.warning(f"  Batch {b_start}–{b_end} failed: {exc}")

    # Interpolate sparse samples to full 0.25° grid using scipy
    from scipy.ndimage import zoom
    if scores.max() > 0:
        # Simple nearest-neighbour fill for unsampled cells
        from scipy.ndimage import generic_filter
        def _fill(arr):
            if arr[len(arr) // 2] == 0 and arr.max() > 0:
                return arr.max()
            return arr[len(arr) // 2]
        scores = generic_filter(scores, _fill, size=step + 1)

    np.save(str(out_path), scores)
    logger.info(
        f"Saved sentinel_scores.npy — shape {scores.shape}, "
        f"mean={scores.mean():.4f}, max={scores.max():.4f}"
    )
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    scores = fetch_sentinel_debris_scores(force_refresh=True)
    print(f"Shape: {scores.shape}  mean: {scores.mean():.4f}  max: {scores.max():.4f}")
