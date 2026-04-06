"""
fetch_cmems.py
Fetches Copernicus Marine (CMEMS) ocean current data using the
copernicusmarine Python SDK.

Datasets fetched
----------------
1. cmems_mod_glo_phy_anfc_0.083deg_P1D-m
   Variables: uo, vo, zos (sea surface height)
   Period: last 30 days + 7-day forecast

2. cmems_mod_glo_phy_my_0.083_P1D-m  (GLOBAL_MULTIYEAR_PHY_001_030)
   Variables: uo, vo
   Period: 1993-01-01 to present — used for GNN training

Output
------
data/cmems_currents.npy     : (N_lat, N_lon, 2) float32 [u, v] m/s  (analysis)
data/cmems_historical.npy   : (T, N_lat, N_lon, 2) float32           (training)

Environment Variables Required
-------------------------------
CMEMS_USERNAME : Copernicus Marine username
CMEMS_PASSWORD : Copernicus Marine password
"""
import os
import logging
import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# Analysis / forecast dataset
ANFC_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
# Historical reanalysis dataset
HIST_DATASET = "cmems_mod_glo_phy_my_0.083_P1D-m"


def _get_credentials() -> tuple[str, str]:
    username = os.environ.get("CMEMS_USERNAME")
    password = os.environ.get("CMEMS_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "CMEMS_USERNAME and CMEMS_PASSWORD must be set in environment."
        )
    return username, password


def _interp_to_025(
    src_lats: np.ndarray, src_lons: np.ndarray,
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation of CMEMS 0.083° data onto 0.25° grid."""
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    tgt_lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat, dtype=np.float64)
    tgt_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float64)

    def _interp(data2d):
        lat_sorted = src_lats if src_lats[0] < src_lats[-1] else src_lats[::-1]
        d_sorted = data2d if src_lats[0] < src_lats[-1] else data2d[::-1, :]
        interp = RegularGridInterpolator(
            (lat_sorted, src_lons), d_sorted,
            method="linear", bounds_error=False, fill_value=0.0
        )
        pts = np.stack(
            np.meshgrid(tgt_lats, tgt_lons, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        return interp(pts).reshape(n_lat, n_lon).astype(np.float32)

    return _interp(u), _interp(v)


def fetch_cmems_analysis() -> np.ndarray:
    """
    Fetch the latest 30-day analysis + 7-day forecast from CMEMS.
    Returns (N_lat, N_lon, 2) float32 [u, v] on 0.25° grid.
    """
    username, password = _get_credentials()
    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError("copernicusmarine SDK not installed. Run: pip install copernicusmarine")

    end_dt   = datetime.utcnow() + timedelta(days=5)
    start_dt = datetime.utcnow() - timedelta(days=10)

    logger.info(f"Fetching CMEMS analysis: {ANFC_DATASET}")
    try:
        ds = cm.open_dataset(
            dataset_id=ANFC_DATASET,
            variables=["uo", "vo"],
            start_datetime=start_dt.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=end_dt.strftime("%Y-%m-%dT00:00:00"),
            minimum_depth=0.0,
            maximum_depth=1.0,
            username=username,
            password=password,
        )

        # Take surface layer mean (squeeze depth if only one level returned)
        uo = ds["uo"]
        vo = ds["vo"]
        if "depth" in uo.dims:
            uo = uo.isel(depth=0)
            vo = vo.isel(depth=0)
        u = uo.mean(dim="time").values
        v = vo.mean(dim="time").values
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        ds.close()

        u_025, v_025 = _interp_to_025(lats, lons, u, v)
        result = np.stack([u_025, v_025], axis=-1)

        out_path = DATA_DIR / "cmems_currents.npy"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), result)
        logger.info(f"Saved cmems_currents.npy — shape {result.shape}")
        return result

    except Exception as exc:
        logger.error(f"CMEMS analysis fetch failed: {exc}")
        n_lat = int(180 / RESOLUTION)
        n_lon = int(360 / RESOLUTION)
        return np.zeros((n_lat, n_lon, 2), dtype=np.float32)


def fetch_cmems_historical(
    start_year: int = 1993,
    end_year: int = 2024,
    sample_every_n_days: int = 30
) -> np.ndarray:
    """
    Fetch historical CMEMS reanalysis for GNN training.
    Returns (T, N_lat, N_lon, 2) float32.
    sample_every_n_days controls temporal density (default: monthly snapshots).
    """
    username, password = _get_credentials()
    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError("copernicusmarine SDK not installed.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = DATA_DIR / "cmems_historical.npy"
    if hist_path.exists():
        logger.info(f"Loading cached CMEMS historical from {hist_path}")
        return np.load(str(hist_path))

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    frames = []

    # Sample one snapshot per month to keep data manageable
    from datetime import date
    snapshots = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            snapshots.append(date(year, month, 15))

    logger.info(f"Fetching {len(snapshots)} historical CMEMS snapshots...")
    for snap in snapshots:
        snap_str = snap.strftime("%Y-%m-%dT00:00:00")
        try:
            ds = cm.open_dataset(
                dataset_id=HIST_DATASET,
                variables=["uo", "vo"],
                start_datetime=snap_str,
                end_datetime=snap_str,
                minimum_depth=0.0,
                maximum_depth=1.0,
                username=username,
                password=password,
            )
            u = ds["uo"].isel(depth=0, time=0).values
            v = ds["vo"].isel(depth=0, time=0).values
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            ds.close()
            u_025, v_025 = _interp_to_025(lats, lons, u, v)
            frames.append(np.stack([u_025, v_025], axis=-1))
        except Exception as exc:
            logger.warning(f"  Skipped {snap}: {exc}")

    if frames:
        result = np.stack(frames, axis=0).astype(np.float32)   # (T, N_lat, N_lon, 2)
        np.save(str(hist_path), result)
        logger.info(f"Saved cmems_historical.npy — shape {result.shape}")
        return result
    else:
        logger.warning("No historical CMEMS data retrieved.")
        return np.zeros((1, n_lat, n_lon, 2), dtype=np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    curr = fetch_cmems_analysis()
    print(f"CMEMS analysis shape: {curr.shape}, max |u|: {np.abs(curr[..., 0]).max():.3f}")
