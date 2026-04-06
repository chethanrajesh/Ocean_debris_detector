"""
fetch_nasa.py
Fetches MODIS Aqua MOD09GA and VIIRS VNP09GA surface reflectance tiles
from NASA Earthdata CMR API, extracts NIR/SWIR bands relevant to
floating debris detection, and saves gridded arrays aligned to the ocean
node graph.

Environment Variables Required
-------------------------------
EARTHDATA_USERNAME : NASA Earthdata login username
EARTHDATA_PASSWORD : NASA Earthdata login password
"""
import os
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
RESOLUTION = 0.25

# CMR concept IDs for the products we need
PRODUCTS = {
    "MOD09GA": {
        "concept_id": "C1621091359-LPDAAC_ECS",   # MODIS Aqua MOD09GA v006
        "nir_band": "sur_refl_b02",
        "swir_band": "sur_refl_b06",
    },
    "VNP09GA": {
        "concept_id": "C1373412048-LPDAAC_ECS",   # VIIRS VNP09GA v001
        "nir_band": "SurfReflect_I2",
        "swir_band": "SurfReflect_I3",
    },
}


def _get_credentials() -> tuple[str, str]:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in environment."
        )
    return username, password


def _search_granules(
    concept_id: str,
    start_date: str,
    end_date: str,
    bbox: str = "-180,-90,180,90",
    page_size: int = 20,
) -> list[dict]:
    """Query CMR for granule download URLs."""
    params = {
        "concept_id": concept_id,
        "temporal": f"{start_date},{end_date}",
        "bounding_box": bbox,
        "page_size": page_size,
        "sort_key": "-start_date",
    }
    resp = requests.get(CMR_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    return entries


def _download_granule(url: str, auth: tuple, dest: Path) -> Path:
    """Stream-download a granule file with Earthdata session authentication."""
    with requests.Session() as session:
        session.auth = auth
        # Earthdata login requires following redirects with auth retained
        resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def _extract_reflectance_to_grid(
    hdf_path: Path, nir_band: str, swir_band: str,
    lats_out: np.ndarray, lons_out: np.ndarray
) -> np.ndarray:
    """
    Extract NIR and SWIR bands from an HDF4/5 file and reproject to the
    0.25° grid. Returns (N_lat, N_lon, 2) array [NIR, SWIR].
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(str(hdf_path), engine="netcdf4")
        nir = ds[nir_band].values.astype(np.float32)
        swir = ds[swir_band].values.astype(np.float32)
        # Scale factor QA: MODIS scale = 0.0001
        nir = np.clip(nir * 0.0001, 0, 1)
        swir = np.clip(swir * 0.0001, 0, 1)
    except Exception:
        # HDF4 fallback
        try:
            from pyhdf.SD import SD, SDC  # type: ignore
            hdf = SD(str(hdf_path), SDC.READ)
            nir_raw = hdf.select(nir_band)[:]
            swir_raw = hdf.select(swir_band)[:]
            nir = np.clip(nir_raw.astype(np.float32) * 0.0001, 0, 1)
            swir = np.clip(swir_raw.astype(np.float32) * 0.0001, 0, 1)
        except Exception as exc:
            logger.error(f"Failed to read {hdf_path}: {exc}")
            n_lat, n_lon = len(lats_out), len(lons_out)
            return np.zeros((n_lat, n_lon, 2), dtype=np.float32)

    # Simple nearest-neighbour reproject to 0.25° grid
    n_lat = len(lats_out)
    n_lon = len(lons_out)
    src_lat = np.linspace(90, -90, nir.shape[-2])
    src_lon = np.linspace(-180, 180, nir.shape[-1])

    out = np.zeros((n_lat, n_lon, 2), dtype=np.float32)
    for i, lat in enumerate(lats_out):
        li = int(np.argmin(np.abs(src_lat - lat)))
        for j, lon in enumerate(lons_out):
            lj = int(np.argmin(np.abs(src_lon - lon)))
            out[i, j, 0] = nir[li, lj] if nir.ndim == 2 else nir[0, li, lj]
            out[i, j, 1] = swir[li, lj] if swir.ndim == 2 else swir[0, li, lj]
    return out


def fetch_nasa_reflectance(days_back: int = 8) -> np.ndarray:
    """
    Download MODIS + VIIRS granules for the last `days_back` days,
    composite NIR and SWIR into a global 0.25° grid.

    Returns
    -------
    reflectance : (N_lat, N_lon, 2) float32 array [NIR, SWIR] in [0, 1]
    """
    auth = _get_credentials()
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end_dt.strftime("%Y-%m-%dT23:59:59Z")

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_out = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat).astype(np.float32)
    lons_out = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon).astype(np.float32)

    composite = np.zeros((n_lat, n_lon, 2), dtype=np.float32)
    count = np.zeros((n_lat, n_lon), dtype=np.int32)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for product, meta in PRODUCTS.items():
        logger.info(f"Searching CMR for {product}...")
        try:
            entries = _search_granules(meta["concept_id"], start_str, end_str, page_size=5)
        except Exception as exc:
            logger.warning(f"CMR search failed for {product}: {exc}")
            continue

        for entry in entries[:3]:  # Limit granules per product for speed
            links = [l["href"] for l in entry.get("links", [])
                     if l.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"]
            if not links:
                continue
            url = links[0]
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / Path(url).name
                try:
                    _download_granule(url, auth, dest)
                    grid = _extract_reflectance_to_grid(
                        dest, meta["nir_band"], meta["swir_band"], lats_out, lons_out
                    )
                    composite += grid
                    count += (grid[..., 0] > 0).astype(np.int32)
                    logger.info(f"  Processed {Path(url).name}")
                except Exception as exc:
                    logger.warning(f"  Skipped {url}: {exc}")

    # Average over granules where data exists
    valid = count > 0
    composite[valid] /= count[valid, np.newaxis]

    out_path = DATA_DIR / "nasa_reflectance.npy"
    np.save(str(out_path), composite)
    logger.info(f"NASA reflectance saved to {out_path} — shape {composite.shape}")
    return composite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arr = fetch_nasa_reflectance(days_back=16)
    print(f"Reflectance shape: {arr.shape}, range: [{arr.min():.4f}, {arr.max():.4f}]")
