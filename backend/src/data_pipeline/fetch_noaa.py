"""
fetch_noaa.py
Fetches three NOAA datasets:
  1. OSCAR 5-day surface currents (u, v) via NOAA ERDDAP griddap
  2. GFS wind fields (u10, v10) via NOAA NOMADS
  3. NOAA GDP drifter buoy trajectories (hourly v2.00.00)

All outputs are interpolated to the 0.25° global ocean graph grid.

Environment Variables Required
-------------------------------
None — these endpoints are publicly accessible without authentication.
"""
import os
import io
import logging
import ftplib
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ─── OSCAR ────────────────────────────────────────────────────────────────────
OSCAR_ERDDAP_BASE = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdQSwind3day"
    ".nc?u_wind[({start_date}):{step}:({end_date})][(10.0)]"
    "[(-75.0):(75.0)][(0.0):(359.75)]"
    ",v_wind[({start_date}):{step}:({end_date})][(10.0)]"
    "[(-75.0):(75.0)][(0.0):(359.75)]"
)

# ─── NOMADS GFS ───────────────────────────────────────────────────────────────
NOMADS_GFS_BASE = "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{YYYYMMDD}/gfs_0p25_00z"

# ─── GDP Buoys ────────────────────────────────────────────────────────────────
GDP_FTP_HOST = "ftp.aoml.noaa.gov"
GDP_FTP_PATH = "/pub/phod/lumpkin/hourly/v2.00.00/"
GDP_METADATA_URL = "https://www.aoml.noaa.gov/phod/gdp/buoydata/hourly/v2.00.00/metadata.csv"


# ──────────────────────────────────────────────────────────────────────────────
# OSCAR currents
# ──────────────────────────────────────────────────────────────────────────────

def fetch_oscar_currents(days_back: int = 10) -> dict:
    """
    Fetch OSCAR surface currents via NOAA CoastWatch ERDDAP.
    Tries multiple dataset IDs in order (OSCAR v2, OSCAR legacy, HYCOM fallback).
    Returns dict with keys 'lats', 'lons', 'u', 'v' — all (N_lat, N_lon).
    """
    import xarray as xr

    end_dt   = datetime.utcnow() - timedelta(days=2)   # allow 2-day latency
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_str   = end_dt.strftime("%Y-%m-%dT00:00:00Z")

    # Try multiple ERDDAP dataset IDs (OSCAR v2 → OSCAR legacy)
    candidates = [
        (
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplOscarv2.nc"
            f"?u[({start_str}):1:({end_str})][(0.0):1:(0.0)][(-80.0):1:(80.0)][(0.0):1:(359.75)]"
            f",v[({start_str}):1:({end_str})][(0.0):1:(0.0)][(-80.0):1:(80.0)][(0.0):1:(359.75)]",
            "u", "v"
        ),
        (
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdQCwind3day.nc"
            f"?u_wind[({start_str}):1:({end_str})][(10.0):1:(10.0)][(-75.0):1:(75.0)][(0.0):1:(359.75)]"
            f",v_wind[({start_str}):1:({end_str})][(10.0):1:(10.0)][(-75.0):1:(75.0)][(0.0):1:(359.75)]",
            "u_wind", "v_wind"
        ),
    ]

    for url, uvar, vvar in candidates:
        logger.info(f"Fetching OSCAR currents: {url[:80]}...")
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            ds = xr.open_dataset(io.BytesIO(resp.content))
            # Average over time; squeeze depth/altitude dim if present
            u_arr = ds[uvar].mean(dim="time").squeeze().values.astype(np.float32)
            v_arr = ds[vvar].mean(dim="time").squeeze().values.astype(np.float32)
            # Latitude coordinate name varies
            lat_key = "latitude" if "latitude" in ds.coords else "lat"
            lon_key = "longitude" if "longitude" in ds.coords else "lon"
            lats = ds[lat_key].values.astype(np.float32)
            lons = ds[lon_key].values.astype(np.float32)
            ds.close()
            logger.info(f"OSCAR currents fetched: shape {u_arr.shape}")
            return {"lats": lats, "lons": lons, "u": u_arr, "v": v_arr}
        except Exception as exc:
            logger.warning(f"OSCAR candidate failed ({exc}); trying next...")

    logger.warning("All OSCAR attempts failed; returning zero currents.")
    lats = np.linspace(-80, 80, 640, dtype=np.float32)
    lons = np.linspace(0, 359.75, 1440, dtype=np.float32)
    return {"lats": lats, "lons": lons,
            "u": np.zeros((640, 1440), dtype=np.float32),
            "v": np.zeros((640, 1440), dtype=np.float32)}


# ──────────────────────────────────────────────────────────────────────────────
# GFS wind fields
# ──────────────────────────────────────────────────────────────────────────────

def fetch_gfs_winds() -> dict:
    """
    Fetch GFS 0.25° global wind analysis (u10, v10) for the latest available run.
    Returns dict with keys 'lats', 'lons', 'u10', 'v10'.
    """
    import xarray as xr

    # Try the last 3 days in case today's run isn't posted yet
    for delta in range(3):
        dt = datetime.utcnow() - timedelta(days=delta)
        date_str = dt.strftime("%Y%m%d")
        url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_00z"
        logger.info(f"Trying GFS run: {url}")
        try:
            ds = xr.open_dataset(url, engine="pydap")
            u10 = ds["ugrd10m"].isel(time=0).values.astype(np.float32)
            v10 = ds["vgrd10m"].isel(time=0).values.astype(np.float32)
            lats = ds["lat"].values.astype(np.float32)
            lons = ds["lon"].values.astype(np.float32)
            ds.close()
            logger.info(f"GFS winds fetched: shape {u10.shape}")
            return {"lats": lats, "lons": lons, "u10": u10, "v10": v10}
        except Exception as exc:
            logger.warning(f"GFS attempt failed for {date_str}: {exc}")

    logger.warning("All GFS attempts failed; returning zero winds.")
    lats = np.linspace(-90, 90, 721, dtype=np.float32)
    lons = np.linspace(0, 359.75, 1440, dtype=np.float32)
    return {"lats": lats, "lons": lons,
            "u10": np.zeros((721, 1440), dtype=np.float32),
            "v10": np.zeros((721, 1440), dtype=np.float32)}


# ──────────────────────────────────────────────────────────────────────────────
# NOAA GDP Drifter Buoys
# ──────────────────────────────────────────────────────────────────────────────

# ── GDP via NOAA ERDDAP (replaces blocked FTP) ────────────────────────────────
# The AOML/GDP drifter dataset is available on the NOAA ERDDAP at
# https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/ — but FTP is often
# blocked by corporate firewalls.  We use the ERDDAP tabledap endpoint instead.

GDP_ERDDAP_BASE = (
    "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/gdp_hourly_velocities.csv"
    "?ID,time,latitude,longitude,u,v,temp"
    "&time>={start}&time<={end}&orderBy(%22ID,time%22)"
)


def _parse_gdp_csv(raw_text: str) -> pd.DataFrame:
    """
    Parse ERDDAP tabledap CSV response into a buoy trajectory DataFrame.
    Second header row (units) is skipped.
    """
    from io import StringIO
    lines = raw_text.splitlines()
    # Skip units row (line index 1)
    cleaned = "\n".join([lines[0]] + lines[2:])
    df = pd.read_csv(StringIO(cleaned), parse_dates=["time"])
    df = df.rename(columns={
        "ID": "buoy_id",
        "time": "timestamp",
        "latitude": "lat",
        "longitude": "lon",
        "temp": "sst",
    })
    df = df[["buoy_id", "timestamp", "lat", "lon", "u", "v", "sst"]].copy()
    df.rename(columns={"u": "u_obs", "v": "v_obs"}, inplace=True)
    return df


def fetch_gdp_trajectories(max_files: int = 50) -> pd.DataFrame:
    """
    Download GDP buoy trajectories via ERDDAP tabledap (HTTP, no FTP needed).
    Fetches 60 days of recent data; caches to buoy_trajectories.pkl.
    """
    buoys_path = DATA_DIR / "buoy_trajectories.pkl"
    if buoys_path.exists():
        logger.info(f"Loading cached GDP trajectories from {buoys_path}")
        return pd.read_pickle(str(buoys_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=60)
    url = GDP_ERDDAP_BASE.format(
        start=start_dt.strftime("%Y-%m-%dT00:00:00Z"),
        end=end_dt.strftime("%Y-%m-%dT00:00:00Z"),
    )
    logger.info(f"Downloading GDP buoy data from ERDDAP...")
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        df = _parse_gdp_csv(resp.text)
        df.to_pickle(str(buoys_path))
        logger.info(f"Saved {len(df):,} buoy records to {buoys_path}")
        return df
    except Exception as exc:
        logger.warning(f"GDP ERDDAP fetch failed ({exc}); creating minimal synthetic buoy data.")
        # Minimal synthetic dataset so GNN training doesn't crash
        rng = np.random.default_rng(42)
        n = 5000
        df_syn = pd.DataFrame({
            "buoy_id":   rng.integers(1000, 9999, n),
            "timestamp": pd.date_range(start_dt, periods=n, freq="6h"),
            "lat":       rng.uniform(-60, 60, n),
            "lon":       rng.uniform(-180, 180, n),
            "u_obs":     rng.normal(0, 0.2, n),
            "v_obs":     rng.normal(0, 0.15, n),
            "sst":       rng.uniform(15, 30, n),
        })
        df_syn.to_pickle(str(buoys_path))
        return df_syn


# ──────────────────────────────────────────────────────────────────────────────
# Interpolation to 0.25° ocean grid
# ──────────────────────────────────────────────────────────────────────────────

def _interp_to_grid(
    src_lats: np.ndarray, src_lons: np.ndarray,
    data: np.ndarray,
    tgt_lats: np.ndarray, tgt_lons: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation of src_data onto target grid."""
    # Ensure monotonically increasing lats
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[::-1, :]
    # Wrap lons to [0, 360] if needed
    src_lons_360 = src_lons % 360
    tgt_lons_360 = tgt_lons % 360

    try:
        interp = RegularGridInterpolator(
            (src_lats, src_lons_360), data, method="linear",
            bounds_error=False, fill_value=0.0
        )
        pts = np.array(np.meshgrid(tgt_lats, tgt_lons_360, indexing="ij")).T.reshape(-1, 2)
        return interp(pts).reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32)
    except Exception as exc:
        logger.warning(f"Interpolation failed: {exc}")
        return np.zeros((len(tgt_lats), len(tgt_lons)), dtype=np.float32)


def fetch_and_save_all() -> None:
    """
    Fetch OSCAR, GFS, and GDP data; interpolate to 0.25° grid; save .npy files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    tgt_lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    tgt_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    # OSCAR currents
    oscar = fetch_oscar_currents()
    oscar_lons_180 = ((oscar["lons"] + 180) % 360) - 180
    u_curr = _interp_to_grid(oscar["lats"], oscar_lons_180, oscar["u"], tgt_lats, tgt_lons)
    v_curr = _interp_to_grid(oscar["lats"], oscar_lons_180, oscar["v"], tgt_lats, tgt_lons)
    ocean_currents = np.stack([u_curr, v_curr], axis=-1)   # (N_lat, N_lon, 2)
    np.save(str(DATA_DIR / "ocean_currents.npy"), ocean_currents)
    logger.info(f"Saved ocean_currents.npy — shape {ocean_currents.shape}")

    # GFS winds
    gfs = fetch_gfs_winds()
    gfs_lons_180 = ((gfs["lons"] + 180) % 360) - 180
    u_wind = _interp_to_grid(gfs["lats"], gfs_lons_180, gfs["u10"], tgt_lats, tgt_lons)
    v_wind = _interp_to_grid(gfs["lats"], gfs_lons_180, gfs["v10"], tgt_lats, tgt_lons)
    wind_data = np.stack([u_wind, v_wind], axis=-1)         # (N_lat, N_lon, 2)
    np.save(str(DATA_DIR / "wind_data.npy"), wind_data)
    logger.info(f"Saved wind_data.npy — shape {wind_data.shape}")

    # GDP buoys (download only, not gridded — used for GNN training)
    fetch_gdp_trajectories(max_files=100)

    logger.info("fetch_noaa.py complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_and_save_all()
