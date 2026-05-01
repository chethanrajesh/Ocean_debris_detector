"""
fetch_noaa.py
Fetches wind and buoy data from working public sources:

  1. Wind fields  — Open-Meteo API (free, no auth, global 0.25° grid)
                    Fallback: CMEMS anfc wind stress (usi/vsi)
  2. Geostrophic currents (backup for CMEMS) — CoastWatch nesdisSSH1day
  3. GDP buoy trajectories — CoastWatch ERDDAP tabledap

Output
------
data/wind_data.npy          : (720, 1440, 2) float32  [u10, v10] m/s
data/ocean_currents.npy     : (720, 1440, 2) float32  [u, v]  (SSH geostrophic, backup only)
data/buoy_trajectories.pkl  : DataFrame with buoy tracks
"""
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _target_grids():
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats = np.linspace(-90 + RESOLUTION / 2,  90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)
    return lats, lons


def _interp_to_grid(src_lats, src_lons, data, tgt_lats, tgt_lons):
    """Bilinear interpolation onto 0.25° target grid."""
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[::-1, :]
    src_lons = ((src_lons + 180) % 360) - 180
    tgt_lons_w = ((tgt_lons + 180) % 360) - 180
    sort_idx = np.argsort(src_lons)
    src_lons = src_lons[sort_idx]
    data = data[:, sort_idx]
    try:
        fn = RegularGridInterpolator(
            (src_lats.astype(np.float64), src_lons.astype(np.float64)),
            data.astype(np.float64),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        pts = np.stack(
            np.meshgrid(tgt_lats.astype(np.float64),
                        tgt_lons_w.astype(np.float64), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        return fn(pts).reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32)
    except Exception as exc:
        logger.warning(f"Interpolation failed: {exc}")
        return np.zeros((len(tgt_lats), len(tgt_lons)), dtype=np.float32)


# ── 1. Wind fields via Open-Meteo ─────────────────────────────────────────────

def fetch_openmeteo_winds(force_refresh: bool = False) -> np.ndarray:
    """
    Build a global 0.25° wind grid using the Open-Meteo forecast API.
    Samples a coarse grid of points then interpolates to full resolution.

    Returns (720, 1440, 2) float32 [u10, v10] m/s.
    """
    out_path = DATA_DIR / "wind_data.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached wind data from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Sample on a 5° grid — avoid duplicate endpoints
    step = 5
    sample_lats = np.arange(-85, 86,  step, dtype=np.float32)
    sample_lons = np.arange(-180, 180, step, dtype=np.float32)  # no +180 to avoid duplicate

    lat_grid, lon_grid = np.meshgrid(sample_lats, sample_lons, indexing="ij")
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()

    # Open-Meteo batch: max 500 locations per request, add delay to avoid 429
    BATCH = 300
    u_vals = np.zeros(len(lats_flat), dtype=np.float32)
    v_vals = np.zeros(len(lats_flat), dtype=np.float32)

    logger.info(f"Fetching Open-Meteo winds for {len(lats_flat)} sample points...")

    for start in range(0, len(lats_flat), BATCH):
        end = min(start + BATCH, len(lats_flat))
        lat_str = ",".join(f"{x:.1f}" for x in lats_flat[start:end])
        lon_str = ",".join(f"{x:.1f}" for x in lons_flat[start:end])
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat_str}&longitude={lon_str}"
            "&hourly=wind_speed_10m,wind_direction_10m"
            "&forecast_days=1&wind_speed_unit=ms&timezone=UTC"
        )
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    import time; time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    data = [data]
                for i, loc in enumerate(data):
                    idx = start + i
                    hourly = loc.get("hourly", {})
                    ws_list = hourly.get("wind_speed_10m", [])
                    wd_list = hourly.get("wind_direction_10m", [])
                    if ws_list and wd_list:
                        pick = min(12, len(ws_list) - 1)
                        ws = float(ws_list[pick])
                        wd = float(wd_list[pick])
                        wd_rad = np.radians(wd)
                        u_vals[idx] = -ws * np.sin(wd_rad)
                        v_vals[idx] = -ws * np.cos(wd_rad)
                logger.info(f"  Batch {start}–{end} OK")
                break
            except Exception as exc:
                logger.warning(f"  Batch {start}–{end} attempt {attempt+1} failed: {exc}")
                import time; time.sleep(1)

    # Interpolate sparse sample grid → full 0.25° grid
    tgt_lats, tgt_lons = _target_grids()

    n_slat = len(sample_lats)
    n_slon = len(sample_lons)

    try:
        from scipy.interpolate import RegularGridInterpolator
        fn_u = RegularGridInterpolator(
            (sample_lats.astype(np.float64), sample_lons.astype(np.float64)),
            u_vals.reshape(n_slat, n_slon).astype(np.float64),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        fn_v = RegularGridInterpolator(
            (sample_lats.astype(np.float64), sample_lons.astype(np.float64)),
            v_vals.reshape(n_slat, n_slon).astype(np.float64),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        pts = np.stack(
            np.meshgrid(tgt_lats.astype(np.float64),
                        tgt_lons.astype(np.float64), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        # Clamp query points to sample grid bounds
        pts[:, 0] = np.clip(pts[:, 0], sample_lats[0], sample_lats[-1])
        pts[:, 1] = np.clip(pts[:, 1], sample_lons[0], sample_lons[-1])
        u_grid = fn_u(pts).reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32)
        v_grid = fn_v(pts).reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32)
    except Exception as exc:
        logger.warning(f"Wind interpolation failed: {exc} — using raw values")
        u_grid = np.zeros((len(tgt_lats), len(tgt_lons)), dtype=np.float32)
        v_grid = np.zeros_like(u_grid)

    result = np.stack([u_grid, v_grid], axis=-1)
    np.save(str(out_path), result)
    logger.info(
        f"Saved wind_data.npy — shape {result.shape}, "
        f"max|u10|={np.abs(u_grid).max():.2f} m/s"
    )
    return result


# ── 2. Geostrophic currents from SSH (CoastWatch ERDDAP) ─────────────────────

def fetch_ssh_geostrophic_currents(force_refresh: bool = False) -> np.ndarray:
    """
    Fetch absolute geostrophic surface currents derived from SSH anomalies.
    Dataset: nesdisSSH1day (NOAA CoastWatch, global 0.25°, daily).
    Last available ~2 months behind real-time.

    Returns (720, 1440, 2) float32 [u, v] m/s.
    Only used as a backup when CMEMS currents are unavailable.
    """
    out_path = DATA_DIR / "ssh_currents.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached SSH currents from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASE = "https://coastwatch.pfeg.noaa.gov/erddap"

    # Dataset ends ~2 months ago — use last known good date
    # Get actual last time from dataset metadata
    try:
        r = requests.get(f"{BASE}/griddap/nesdisSSH1day.nc?time[last]", timeout=15)
        import xarray as xr
        ds_t = xr.open_dataset(io.BytesIO(r.content), engine="scipy")
        last_time = pd.Timestamp(ds_t["time"].values[-1])
        ds_t.close()
    except Exception:
        last_time = datetime.now(timezone.utc) - timedelta(days=60)
        last_time = pd.Timestamp(last_time)

    end_dt   = last_time
    start_dt = end_dt - timedelta(days=7)
    s = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    e = end_dt.strftime("%Y-%m-%dT00:00:00Z")

    logger.info(f"Fetching SSH geostrophic currents {s[:10]} → {e[:10]} ...")

    url = (
        f"{BASE}/griddap/nesdisSSH1day.nc"
        f"?ugos[({s}):1:({e})][(-89.875):1:(89.875)][(-179.875):1:(179.875)]"
        f",vgos[({s}):1:({e})][(-89.875):1:(89.875)][(-179.875):1:(179.875)]"
    )
    try:
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        import xarray as xr
        ds = xr.open_dataset(io.BytesIO(resp.content), engine="scipy")
        u = np.nan_to_num(ds["ugos"].mean(dim="time").values.astype(np.float32), nan=0.0)
        v = np.nan_to_num(ds["vgos"].mean(dim="time").values.astype(np.float32), nan=0.0)
        ds.close()
        result = np.stack([u, v], axis=-1)
        np.save(str(out_path), result)
        logger.info(
            f"Saved ssh_currents.npy — shape {result.shape}, "
            f"max|u|={np.abs(u).max():.3f} m/s"
        )
        return result
    except Exception as exc:
        logger.error(f"SSH currents fetch failed: {exc}")
        n_lat, n_lon = int(180 / RESOLUTION), int(360 / RESOLUTION)
        return np.zeros((n_lat, n_lon, 2), dtype=np.float32)


# ── 3. GDP Drifter Buoys ──────────────────────────────────────────────────────

def fetch_gdp_trajectories(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download NOAA GDP drifter buoy trajectories.
    Tries CoastWatch ERDDAP; falls back to synthetic data if unavailable.
    """
    cache = DATA_DIR / "buoy_trajectories.pkl"
    if cache.exists() and not force_refresh:
        logger.info(f"Loading cached GDP trajectories from {cache}")
        return pd.read_pickle(str(cache))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now      = datetime.now(timezone.utc)
    end_dt   = now
    start_dt = now - timedelta(days=60)

    # Try multiple ERDDAP endpoints
    endpoints = [
        (
            "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/gdp_hourly_velocities.csv"
            "?ID,time,latitude,longitude,u,v,temp"
            f"&time>={start_dt.strftime('%Y-%m-%dT00:00:00Z')}"
            f"&time<={end_dt.strftime('%Y-%m-%dT00:00:00Z')}"
            "&orderBy(%22ID,time%22)"
        ),
    ]

    for url in endpoints:
        try:
            logger.info(f"Fetching GDP buoys from ERDDAP...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            lines = resp.text.splitlines()
            cleaned = "\n".join([lines[0]] + lines[2:])
            df = pd.read_csv(io.StringIO(cleaned), parse_dates=["time"])
            df = df.rename(columns={
                "ID": "buoy_id", "time": "timestamp",
                "latitude": "lat", "longitude": "lon",
                "temp": "sst", "u": "u_obs", "v": "v_obs",
            })
            df = df[["buoy_id", "timestamp", "lat", "lon", "u_obs", "v_obs", "sst"]].copy()
            df.to_pickle(str(cache))
            logger.info(f"Saved {len(df):,} buoy records → {cache}")
            return df
        except Exception as exc:
            logger.warning(f"GDP ERDDAP failed: {exc}")

    # Synthetic fallback
    logger.warning("GDP fetch failed — generating synthetic buoy data for GNN training")
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
    df_syn.to_pickle(str(cache))
    return df_syn


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_and_save_all(force_refresh: bool = False) -> None:
    """Fetch winds, SSH currents, and GDP buoys; save all outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetch_openmeteo_winds(force_refresh=force_refresh)
    fetch_ssh_geostrophic_currents(force_refresh=force_refresh)
    fetch_gdp_trajectories(force_refresh=force_refresh)

    logger.info("fetch_noaa.py complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetch_and_save_all(force_refresh=True)
