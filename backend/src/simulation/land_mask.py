"""
land_mask.py
Generates and saves the global ocean land mask at 0.25° resolution.

The mask is derived from ETOPO1 bathymetry: cells with elevation <= 0 m
(i.e. at or below sea level) are classified as ocean.

Output
------
data/land_mask.npy : shape (N_lat, N_lon) boolean array
                     True  = ocean node
                     False = land node
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

RESOLUTION = 0.25  # degrees
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MASK_PATH = os.path.join(DATA_DIR, "land_mask.npy")


def build_land_mask_from_etopo(etopo_z: np.ndarray) -> np.ndarray:
    """
    Given a 2-D array of ETOPO1 elevations (metres), return a boolean
    ocean mask: True where elevation <= 0 (ocean / sea floor).
    """
    return etopo_z <= 0


def build_default_ocean_mask(n_lat: int, n_lon: int) -> np.ndarray:
    """
    Fallback mask when ETOPO1 is unavailable: approximates ocean using
    a simple latitude-band heuristic (excludes obvious polar land masses).
    NOT suitable for production — triggers a warning.
    """
    logger.warning(
        "ETOPO1 data unavailable — using approximate fallback ocean mask. "
        "Run fetch_sentinel.py with GEE credentials to obtain the full mask."
    )
    mask = np.ones((n_lat, n_lon), dtype=bool)

    lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat)
    lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon)

    # Rough land exclusion polygons (lat_min, lat_max, lon_min, lon_max)
    land_boxes = [
        # North America
        (24, 72, -168, -52),
        # South America
        (-56, 12, -82, -34),
        # Europe
        (36, 71, -10, 40),
        # Africa
        (-35, 37, -18, 51),
        # Asia
        (0, 77, 26, 145),
        # Australia
        (-44, -10, 113, 154),
        # Antarctica
        (-90, -60, -180, 180),
        # Greenland
        (60, 84, -55, -18),
    ]

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    for lat_min, lat_max, lon_min, lon_max in land_boxes:
        in_box = (
            (lat_grid >= lat_min) & (lat_grid <= lat_max) &
            (lon_grid >= lon_min) & (lon_grid <= lon_max)
        )
        mask[in_box] = False

    return mask


def generate_mask(force_rebuild: bool = False) -> np.ndarray:
    """
    Load or generate the global ocean mask.

    1. If MASK_PATH exists and force_rebuild=False, load and return it.
    2. Try to load ETOPO1 via GEE (requires credentials) — timeout 20 s.
    3. Fall back to the approximate heuristic mask.

    Returns
    -------
    mask : (N_lat, N_lon) boolean ndarray
    """
    if os.path.exists(MASK_PATH) and not force_rebuild:
        logger.info(f"Loading existing land mask from {MASK_PATH}")
        return np.load(MASK_PATH)

    os.makedirs(DATA_DIR, exist_ok=True)

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)

    # ── Attempt GEE-based ETOPO1 retrieval inside a thread with timeout ────────
    result_holder: dict = {}

    def _gee_fetch():
        try:
            import ee
            from dotenv import load_dotenv
            load_dotenv()

            key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "")
            email    = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL", "")
            if not key_path or not os.path.exists(key_path):
                raise EnvironmentError("GEE_SERVICE_ACCOUNT_KEY not set or file not found.")

            credentials = ee.ServiceAccountCredentials(email or None, key_path)
            ee.Initialize(credentials, opt_url="https://earthengine.googleapis.com")
            logger.info("GEE initialized for ETOPO1 retrieval.")

            lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat)
            lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon)

            etopo = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock")
            # Coarse sample: every 1° grid point (~64 k points → manageable)
            points = [
                ee.Feature(ee.Geometry.Point([float(lon), float(lat)]),
                           {"lat": float(lat), "lon": float(lon)})
                for lat in lats[::4] for lon in lons[::4]
            ]
            fc      = ee.FeatureCollection(points)
            sampled = etopo.sampleRegions(collection=fc, scale=25000, geometries=False)
            values  = sampled.getInfo()

            elev_dict: dict = {}
            for f in values["features"]:
                p = f["properties"]
                elev_dict[(round(p["lat"], 2), round(p["lon"], 2))] = p.get("bedrock", 0)

            elev_grid = np.zeros((n_lat, n_lon), dtype=np.float32)
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    key = (round(float(lat), 2), round(float(lon), 2))
                    elev_grid[i, j] = elev_dict.get(key, 0.0)

            result_holder["mask"] = build_land_mask_from_etopo(elev_grid)
            logger.info("ETOPO1-based mask built successfully.")
        except Exception as exc:
            result_holder["error"] = str(exc)
            logger.warning(f"GEE ETOPO1 retrieval failed: {exc}")

    import threading
    t = threading.Thread(target=_gee_fetch, daemon=True)
    t.start()
    t.join(timeout=25)   # wait max 25 seconds

    if "mask" in result_holder:
        mask = result_holder["mask"]
    else:
        reason = result_holder.get("error", "timeout after 25 s")
        logger.warning(f"GEE not used ({reason}); falling back to heuristic ocean mask.")
        mask = build_default_ocean_mask(n_lat, n_lon)

    np.save(MASK_PATH, mask)
    logger.info(f"Land mask saved to {MASK_PATH} — ocean nodes: {mask.sum():,}")
    return mask



def get_ocean_node_coordinates(mask: np.ndarray):
    """
    Return (lats, lons) arrays for all ocean nodes, and the flat ocean index mapping.

    Returns
    -------
    ocean_lats : (N_ocean,) float32
    ocean_lons : (N_ocean,) float32
    node_id    : (N_lat, N_lon) int32  — -1 for land, >=0 for ocean node index
    """
    n_lat, n_lon = mask.shape
    all_lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat).astype(np.float32)
    all_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon).astype(np.float32)

    node_id = np.full((n_lat, n_lon), -1, dtype=np.int32)
    ocean_lats, ocean_lons = [], []
    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            if mask[i, j]:
                node_id[i, j] = idx
                ocean_lats.append(all_lats[i])
                ocean_lons.append(all_lons[j])
                idx += 1

    return np.array(ocean_lats), np.array(ocean_lons), node_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mask = generate_mask(force_rebuild=True)
    lats, lons, nid = get_ocean_node_coordinates(mask)
    print(f"Ocean nodes: {len(lats):,} / Total grid: {mask.size:,}")
