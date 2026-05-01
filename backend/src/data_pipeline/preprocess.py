"""
preprocess.py
Aligns all fetched datasets to the 0.25° global ocean grid,
applies the land mask, and saves processed arrays to data/.

Pipeline
--------
1. Load land mask
2. Load CMEMS currents (primary) — blend with SSH geostrophic if available
3. Load Open-Meteo wind fields
4. Compute Stokes drift magnitude
5. Load / generate initial density map from Sentinel-2 scores
6. Seed known pollution source points
7. Build ocean node graph  →  global_graph/nodes.npy
8. Save source_points.npy, cleanup_routes.npy, density_maps.npy

Output files
------------
data/global_graph/nodes.npy  : (N_ocean, 7) [lat, lon, u, v, u_wind, v_wind, mask_bit]
data/density_maps.npy        : (720, 1440)  initial debris density proxy
data/source_points.npy       : (N_src, 2)  [lat, lon]
data/cleanup_routes.npy      : (N_src, 3)  [lat, lon, priority]
data/stokes_drift.npy        : (720, 1440)  Stokes drift magnitude m/s
"""
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent.parent.parent / "data"
GRAPH_DIR = DATA_DIR / "global_graph"
RESOLUTION = 0.25

# Known plastic pollution source regions (Lebreton 2017, Jambeck 2015)
KNOWN_SOURCES = [
    (30.0, 121.5, "Yangtze River Delta"),
    (23.0, 113.5, "Pearl River Delta"),
    (22.0, 114.0, "Hong Kong coastal"),
    (35.5, 139.8, "Tokyo Bay"),
    (37.5, 126.8, "Han River / Seoul coast"),
    (10.0, 105.0, "Mekong Delta"),
    (16.0, 108.0, "Vietnam central coast"),
    (23.7,  90.4, "Ganges/Brahmaputra Delta"),
    (13.0,  80.3, "Chennai coast"),
    (19.1,  72.8, "Mumbai coast"),
    ( 6.9,  79.8, "Colombo coast"),
    ( 3.1, 101.5, "Klang River, Malaysia"),
    (-6.2, 106.8, "Jakarta coast"),
    (-7.2, 112.7, "Surabaya coast"),
    ( 5.3,  -4.0, "Abidjan coast"),
    ( 6.4,   3.4, "Lagos Bight"),
    (-4.3,  15.3, "Congo River mouth"),
    (-25.9, 32.6, "Maputo coast"),
    (18.5, -69.9, "Dominican Republic coast"),
    (23.1, -82.4, "Havana coastal"),
    (-23.0,-43.2, "Rio de Janeiro / Guanabara Bay"),
    (10.5, -66.9, "Caracas coastal"),
    (43.3,   5.4, "Marseille coastal"),
    (37.9,  23.7, "Athens / Saronic Gulf"),
    (38.2,  15.6, "Messina Strait"),
]

GYRE_CENTRES = [(30, -140), (-30, -100), (30, 160), (-30, 80)]


def _normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if (mx - mn) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _latlon_to_idx(lat, lon, n_lat, n_lon):
    i = int(round((lat + 90 - RESOLUTION / 2) / RESOLUTION))
    j = int(round((lon + 180 - RESOLUTION / 2) / RESOLUTION))
    return max(0, min(n_lat - 1, i)), max(0, min(n_lon - 1, j))


def run_preprocessing() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    all_lats = np.linspace(-90 + RESOLUTION / 2,  90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    all_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    # ── 1. Land mask ──────────────────────────────────────────────────────────
    mask_path = DATA_DIR / "land_mask.npy"
    if mask_path.exists():
        land_mask = np.load(str(mask_path)).astype(bool)
        logger.info(f"Land mask: {land_mask.sum():,} ocean cells")
    else:
        logger.warning("Land mask missing — generating...")
        from src.simulation.land_mask import generate_mask
        land_mask = generate_mask()

    # ── 2. Ocean currents — CMEMS primary, SSH geostrophic backup ─────────────
    cmems_path = DATA_DIR / "cmems_currents.npy"
    ssh_path   = DATA_DIR / "ssh_currents.npy"
    oscar_path = DATA_DIR / "ocean_currents.npy"

    if cmems_path.exists():
        currents = np.load(str(cmems_path))
        logger.info(f"Using CMEMS currents — max|u|={np.abs(currents[...,0]).max():.3f}")
        # Blend with SSH geostrophic if available (adds mesoscale eddies)
        if ssh_path.exists():
            ssh = np.load(str(ssh_path))
            currents = 0.7 * currents + 0.3 * ssh
            logger.info("Blended CMEMS (70%) + SSH geostrophic (30%)")
    elif ssh_path.exists():
        currents = np.load(str(ssh_path))
        logger.info(f"Using SSH geostrophic currents — max|u|={np.abs(currents[...,0]).max():.3f}")
    elif oscar_path.exists():
        currents = np.load(str(oscar_path))
        logger.info(f"Using OSCAR currents — max|u|={np.abs(currents[...,0]).max():.3f}")
    else:
        logger.warning("No current data found — using zeros")
        currents = np.zeros((n_lat, n_lon, 2), dtype=np.float32)

    currents = np.nan_to_num(currents, nan=0.0)
    u_curr = currents[..., 0]
    v_curr = currents[..., 1]
    u_curr[~land_mask] = 0.0
    v_curr[~land_mask] = 0.0

    # Save blended result as canonical ocean_currents.npy
    np.save(str(oscar_path), np.stack([u_curr, v_curr], axis=-1))

    # ── 3. Wind fields ────────────────────────────────────────────────────────
    wind_path = DATA_DIR / "wind_data.npy"
    if wind_path.exists():
        winds = np.nan_to_num(np.load(str(wind_path)), nan=0.0)
        logger.info(f"Wind data loaded — max|u10|={np.abs(winds[...,0]).max():.2f}")
    else:
        logger.warning("wind_data.npy missing — using zeros")
        winds = np.zeros((n_lat, n_lon, 2), dtype=np.float32)

    u_wind = winds[..., 0]
    v_wind = winds[..., 1]
    u_wind[~land_mask] = 0.0
    v_wind[~land_mask] = 0.0

    # ── 4. Stokes drift ───────────────────────────────────────────────────────
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    stokes_mag = np.clip(0.012 * wind_speed**2 / 9.81, 0, 0.5)
    stokes_mag[~land_mask] = 0.0
    np.save(str(DATA_DIR / "stokes_drift.npy"), stokes_mag.astype(np.float32))
    logger.info(f"Stokes drift — max={stokes_mag.max():.4f} m/s")

    # ── 5. Initial density from Sentinel-2 ───────────────────────────────────
    s2_path = DATA_DIR / "sentinel_scores.npy"
    if s2_path.exists():
        sentinel_raw = np.load(str(s2_path))
        if sentinel_raw.shape != (n_lat, n_lon):
            pad = np.zeros((n_lat, n_lon), dtype=np.float32)
            r = min(sentinel_raw.shape[0], n_lat)
            c = min(sentinel_raw.shape[1], n_lon)
            pad[:r, :c] = sentinel_raw[:r, :c]
            sentinel_raw = pad
        density = sentinel_raw.astype(np.float32)
        logger.info(f"Sentinel-2 scores loaded — max={density.max():.4f}")
    else:
        logger.warning("sentinel_scores.npy missing — seeding from source points only")
        density = np.zeros((n_lat, n_lon), dtype=np.float32)

    # ── 6. Seed known source points + AOML debris seeds ──────────────────────
    source_coords = []

    # First seed from AOML debris concentration data
    aoml_path = DATA_DIR / "debris_seed_nodes.npy"
    if aoml_path.exists():
        aoml_seeds = np.load(str(aoml_path))  # (N, 3) [lat, lon, conc_norm]
        logger.info(f"Seeding density from {len(aoml_seeds)} AOML nodes...")
        for lat, lon, conc in aoml_seeds:
            i, j = _latlon_to_idx(float(lat), float(lon), n_lat, n_lon)
            if land_mask[i, j]:
                density[i, j] = max(density[i, j], float(conc))
                source_coords.append([all_lats[i], all_lons[j]])

    # Then seed from known pollution source points
    for lat, lon, name in KNOWN_SOURCES:
        i, j = _latlon_to_idx(lat, lon, n_lat, n_lon)
        if land_mask[i, j]:
            density[i, j] = max(density[i, j], 0.6)
            source_coords.append([all_lats[i], all_lons[j]])
        else:
            # Find nearest ocean cell
            for di in range(-5, 6):
                found = False
                for dj in range(-5, 6):
                    ni, nj = i + di, (j + dj) % n_lon
                    if 0 <= ni < n_lat and land_mask[ni, nj]:
                        density[ni, nj] = max(density[ni, nj], 0.5)
                        source_coords.append([all_lats[ni], all_lons[nj]])
                        found = True
                        break
                if found:
                    break
        logger.debug(f"  Seeded: {name}")

    source_arr = np.array(source_coords, dtype=np.float32)
    np.save(str(DATA_DIR / "source_points.npy"), source_arr)
    logger.info(f"source_points.npy: {len(source_arr)} points")

    # Smooth and normalise density
    density = gaussian_filter(density, sigma=1.5)
    density[~land_mask] = 0.0
    density = _normalize(density)
    np.save(str(DATA_DIR / "density_maps.npy"), density.astype(np.float32))
    logger.info(f"density_maps.npy — range [{density.min():.3f}, {density.max():.3f}]")

    # ── 7. Cleanup routes ─────────────────────────────────────────────────────
    cleanup = []
    top_sources = source_arr[:20] if len(source_arr) >= 20 else source_arr
    for lat, lon in top_sources:
        min_dist = float("inf")
        for glat, glon in GYRE_CENTRES:
            dist = ((lat - glat)**2 + (lon - glon)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
        priority = float(np.clip(1.0 - min_dist / 180.0, 0, 1))
        cleanup.append([lat, lon, priority])
    np.save(str(DATA_DIR / "cleanup_routes.npy"), np.array(cleanup, dtype=np.float32))
    logger.info(f"cleanup_routes.npy: {len(cleanup)} routes")

    # ── 8. Ocean node graph ───────────────────────────────────────────────────
    logger.info("Building ocean node graph...")
    ocean_rows, ocean_cols = np.where(land_mask)
    N = len(ocean_rows)

    nodes = np.zeros((N, 7), dtype=np.float32)
    nodes[:, 0] = all_lats[ocean_rows]
    nodes[:, 1] = all_lons[ocean_cols]
    nodes[:, 2] = u_curr[ocean_rows, ocean_cols]
    nodes[:, 3] = v_curr[ocean_rows, ocean_cols]
    nodes[:, 4] = u_wind[ocean_rows, ocean_cols]
    nodes[:, 5] = v_wind[ocean_rows, ocean_cols]
    nodes[:, 6] = 1.0

    # Normalise current and wind columns
    for col in [2, 3, 4, 5]:
        mn, mx = nodes[:, col].min(), nodes[:, col].max()
        if mx - mn > 1e-8:
            nodes[:, col] = (nodes[:, col] - mn) / (mx - mn)

    np.save(str(GRAPH_DIR / "nodes.npy"), nodes)
    logger.info(f"nodes.npy — shape {nodes.shape}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_preprocessing()
