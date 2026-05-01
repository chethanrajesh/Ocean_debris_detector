"""
fetch_cmems.py
Fetches Copernicus Marine (CMEMS) ocean current data using the
copernicusmarine Python SDK v2.x.

Datasets fetched
----------------
cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m
  Variables: uo, vo  (surface u/v currents, m/s)
  Period: last 5 days (analysis + short forecast)

Output
------
data/cmems_currents.npy  : (720, 1440, 2) float32  [u, v] on 0.25° grid
data/ocean_currents.npy  : same — canonical name read by the rest of the pipeline
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

DATASET_ID = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"


def _get_credentials() -> tuple[str, str]:
    username = os.environ.get("CMEMS_USERNAME", "")
    password = os.environ.get("CMEMS_PASSWORD", "")
    if not username or not password:
        raise EnvironmentError("CMEMS_USERNAME and CMEMS_PASSWORD must be set in .env")
    return username, password


def _interp_to_025(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation from CMEMS 0.083° grid onto 0.25° grid."""
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    tgt_lats = np.linspace(-90 + RESOLUTION / 2,  90 - RESOLUTION / 2, n_lat, dtype=np.float64)
    tgt_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float64)

    # Ensure lats are ascending
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        u = u[::-1, :]
        v = v[::-1, :]

    def _interp_one(data2d: np.ndarray) -> np.ndarray:
        fn = RegularGridInterpolator(
            (src_lats.astype(np.float64), src_lons.astype(np.float64)),
            data2d.astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.stack(
            np.meshgrid(tgt_lats, tgt_lons, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        return fn(pts).reshape(n_lat, n_lon).astype(np.float32)

    return _interp_one(u), _interp_one(v)


def fetch_cmems_currents(force_refresh: bool = False) -> np.ndarray:
    """
    Download latest CMEMS surface currents and save to data/.

    Returns (720, 1440, 2) float32 [u, v] on 0.25° grid.
    """
    out_path = DATA_DIR / "cmems_currents.npy"
    ocean_path = DATA_DIR / "ocean_currents.npy"

    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached CMEMS currents from {out_path}")
        arr = np.load(str(out_path))
        if not ocean_path.exists():
            np.save(str(ocean_path), arr)
        return arr

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    username, password = _get_credentials()

    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError("Run: pip install copernicusmarine")

    now      = datetime.now(timezone.utc)
    end_dt   = now
    start_dt = now - timedelta(days=5)

    logger.info(f"Fetching CMEMS currents {start_dt.date()} → {end_dt.date()} ...")

    ds = cm.open_dataset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime=start_dt.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=end_dt.strftime("%Y-%m-%dT00:00:00"),
        username=username,
        password=password,
    )

    # Take surface layer, average over available time steps
    uo = ds["uo"]
    vo = ds["vo"]
    if "depth" in uo.dims:
        uo = uo.isel(depth=0)
        vo = vo.isel(depth=0)
    u = uo.mean(dim="time").values.astype(np.float32)
    v = vo.mean(dim="time").values.astype(np.float32)
    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)
    ds.close()

    logger.info(f"  Raw shape: u={u.shape}, lat range [{lats.min():.1f}, {lats.max():.1f}]")

    # Replace NaN (land/missing) with 0 before interpolation
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    u_025, v_025 = _interp_to_025(lats, lons, u, v)
    result = np.stack([u_025, v_025], axis=-1)  # (720, 1440, 2)

    np.save(str(out_path),   result)
    np.save(str(ocean_path), result)
    logger.info(
        f"Saved cmems_currents.npy + ocean_currents.npy — shape {result.shape}, "
        f"max|u|={float(np.abs(result[..., 0]).max()):.3f} m/s"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    arr = fetch_cmems_currents(force_refresh=True)
    print(f"Shape: {arr.shape}  max|u|={np.abs(arr[...,0]).max():.3f}  max|v|={np.abs(arr[...,1]).max():.3f}")
