"""
preprocess.py
Aligns all fetched datasets to the 0.25° × 0.25° global ocean grid,
applies the land mask, normalizes variables to [0, 1], computes Stokes
drift magnitude, and saves all processed arrays to data/.

Output files
------------
data/global_graph/nodes.npy    : (N, 7)  [lat, lon, u, v, u_wind, v_wind, mask_bit]
data/density_maps.npy          : (N_lat, N_lon)  initial surface reflectance proxy
data/source_points.npy         : (N_src, 2)  [lat, lon] known river/coastal source nodes
data/cleanup_routes.npy        : (N_src, 3)  [lat, lon, priority]
"""
import logging
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
GRAPH_DIR = DATA_DIR / "global_graph"
RESOLUTION = 0.25

# ─── Major oceanic plastic source regions (river mouths and coastal zones) ────
# Based on Lebreton et al. (2017) and Jambeck et al. (2015)
# Format: (lat, lon, source_name)
KNOWN_SOURCES = [
    # East Asia
    (30.0, 121.5, "Yangtze River Delta"),
    (23.0, 113.5, "Pearl River Delta"),
    (22.0, 114.0, "Hong Kong coastal"),
    (35.5, 139.8, "Tokyo Bay"),
    (37.5, 126.8, "Han River / Seoul coast"),
    (10.0, 105.0, "Mekong Delta"),
    (16.0, 108.0, "Vietnam central coast"),
    # South/Southeast Asia
    (23.7,  90.4, "Ganges/Brahmaputra Delta"),
    (13.0,  80.3, "Chennai coast"),
    (19.1,  72.8, "Mumbai coast"),
    (6.9,  79.8, "Colombo coast"),
    (3.1,  101.5, "Klang River, Malaysia"),
    (-6.2, 106.8, "Jakarta coast"),
    (-7.2, 112.7, "Surabaya coast"),
    # Africa
    (5.3,  -4.0, "Abidjan coast"),
    (6.4,   3.4, "Lagos Bight"),
    (-4.3,  15.3, "Congo River mouth"),
    (-25.9,  32.6, "Maputo coast"),
    # Americas
    (18.5, -69.9, "Dominican Republic coast"),
    (23.1, -82.4, "Havana coastal"),
    (-23.0, -43.2, "Rio de Janeiro / Guanabara Bay"),
    (-33.5, -70.6, "Santiago/Maipo River"),
    (10.5, -66.9, "Caracas coastal"),
    # Mediterranean / Europe
    (43.3,   5.4, "Marseille coastal"),
    (37.9,  23.7, "Athens / Saronic Gulf"),
    (38.2,  15.6, "Messina Strait"),
]


def _normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize array to [0, 1] using min-max."""
    mn, mx = arr.min(), arr.max()
    if (mx - mn) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _latlon_to_idx(lat: float, lon: float, n_lat: int, n_lon: int):
    """Convert a lat/lon to nearest grid index."""
    i = int(round((lat + 90 - RESOLUTION / 2) / RESOLUTION))
    j = int(round((lon + 180 - RESOLUTION / 2) / RESOLUTION))
    i = max(0, min(n_lat - 1, i))
    j = max(0, min(n_lon - 1, j))
    return i, j


def run_preprocessing() -> None:
    """Main preprocessing pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    all_lats = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    all_lons = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    # ── 1. Load land mask ──────────────────────────────────────────────────────
    mask_path = DATA_DIR / "land_mask.npy"
    if mask_path.exists():
        land_mask = np.load(str(mask_path))
        logger.info(f"Loaded land mask: {land_mask.sum():,} ocean nodes")
    else:
        logger.warning("Land mask not found — generating default...")
        from src.simulation.land_mask import generate_mask
        land_mask = generate_mask()

    # ── 2. Load current vectors ────────────────────────────────────────────────
    curr_path = DATA_DIR / "ocean_currents.npy"
    if curr_path.exists():
        currents = np.load(str(curr_path))   # (N_lat, N_lon, 2)
    else:
        logger.warning("ocean_currents.npy missing — using zeros.")
        currents = np.zeros((n_lat, n_lon, 2), dtype=np.float32)

    u_curr = currents[..., 0]
    v_curr = currents[..., 1]

    # Merge CMEMS if available
    cmems_path = DATA_DIR / "cmems_currents.npy"
    if cmems_path.exists():
        cmems = np.load(str(cmems_path))
        logger.info("Blending CMEMS currents with OSCAR (70% CMEMS, 30% OSCAR)...")
        u_curr = 0.7 * cmems[..., 0] + 0.3 * u_curr
        v_curr = 0.7 * cmems[..., 1] + 0.3 * v_curr

    # Apply land mask
    u_curr[~land_mask] = 0.0
    v_curr[~land_mask] = 0.0

    # ── 3. Load wind vectors ───────────────────────────────────────────────────
    wind_path = DATA_DIR / "wind_data.npy"
    if wind_path.exists():
        winds = np.load(str(wind_path))
    else:
        logger.warning("wind_data.npy missing — using zeros.")
        winds = np.zeros((n_lat, n_lon, 2), dtype=np.float32)

    u_wind = winds[..., 0]
    v_wind = winds[..., 1]
    u_wind[~land_mask] = 0.0
    v_wind[~land_mask] = 0.0

    # ── 4. Compute Stokes drift magnitude (approximation) ──────────────────────
    # Stokes = 0.012 × |wind|^2 / (g × wave_period) — simplified parametrization
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    stokes_mag = 0.012 * (wind_speed ** 2) / 9.81
    stokes_mag = np.clip(stokes_mag, 0, 0.5)
    stokes_mag[~land_mask] = 0.0
    np.save(str(DATA_DIR / "stokes_drift.npy"), stokes_mag.astype(np.float32))

    # ── 5. Load or generate initial density map (Sentinel-2 proxy) ────────────
    s2_path = DATA_DIR / "sentinel_scores.npy"
    if s2_path.exists():
        sentinel_raw = np.load(str(s2_path))   # (N_lat, N_lon)
        if sentinel_raw.shape != (n_lat, n_lon):
            # Pad/crop to match grid
            pad_arr = np.zeros((n_lat, n_lon), dtype=np.float32)
            r = min(sentinel_raw.shape[0], n_lat)
            c = min(sentinel_raw.shape[1], n_lon)
            pad_arr[:r, :c] = sentinel_raw[:r, :c]
            sentinel_raw = pad_arr
        density = sentinel_raw.astype(np.float32)
    else:
        logger.warning("sentinel_scores.npy missing — seeding density from source points.")
        density = np.zeros((n_lat, n_lon), dtype=np.float32)

    # ── 6. Seed source points ─────────────────────────────────────────────────
    source_coords = []
    for lat, lon, name in KNOWN_SOURCES:
        i, j = _latlon_to_idx(lat, lon, n_lat, n_lon)
        if land_mask[i, j]:
            # Source is on ocean — seed density
            density[i, j] = max(density[i, j], 0.6)
            source_coords.append([all_lats[i], all_lons[j]])
        else:
            # Nearest ocean neighbor search
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    ni, nj = i + di, (j + dj) % n_lon
                    if 0 <= ni < n_lat and land_mask[ni, nj]:
                        density[ni, nj] = max(density[ni, nj], 0.5)
                        source_coords.append([all_lats[ni], all_lons[nj]])
                        break
                else:
                    continue
                break
        logger.info(f"  Source seeded: {name} @ ({lat}, {lon})")

    source_arr = np.array(source_coords, dtype=np.float32)
    np.save(str(DATA_DIR / "source_points.npy"), source_arr)
    logger.info(f"source_points.npy: {len(source_arr)} points")

    # Smooth density with Gaussian filter
    density = gaussian_filter(density, sigma=1.5)
    density[~land_mask] = 0.0
    density = _normalize(density)
    np.save(str(DATA_DIR / "density_maps.npy"), density.astype(np.float32))
    logger.info(f"density_maps.npy saved — range [{density.min():.3f}, {density.max():.3f}]")

    # ── 7. Build cleanup routes (highest density source → open ocean) ──────────
    # Simple heuristic: top-N density source nodes, sorted by proximity to gyre centres
    GYRE_CENTRES = [(30, -140), (-30, -100), (30, 160), (-30, 80)]
    cleanup = []
    top_sources = source_arr[:20] if len(source_arr) >= 20 else source_arr
    for lat, lon in top_sources:
        min_dist = float("inf")
        best_gyre = GYRE_CENTRES[0]
        for glat, glon in GYRE_CENTRES:
            dist = ((lat - glat)**2 + (lon - glon)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_gyre = (glat, glon)
        priority = float(np.clip(1.0 - min_dist / 180.0, 0, 1))
        cleanup.append([lat, lon, priority])
    np.save(str(DATA_DIR / "cleanup_routes.npy"), np.array(cleanup, dtype=np.float32))
    logger.info(f"cleanup_routes.npy: {len(cleanup)} routes")

    # ── 8. Build ocean node graph ──────────────────────────────────────────────
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
    nodes[:, 6] = 1.0   # land_mask_bit = 1 (ocean)

    # Normalize current and wind columns in-place
    for col in [2, 3, 4, 5]:
        mn, mx = nodes[:, col].min(), nodes[:, col].max()
        if mx - mn > 1e-8:
            nodes[:, col] = (nodes[:, col] - mn) / (mx - mn)

    np.save(str(GRAPH_DIR / "nodes.npy"), nodes)
    logger.info(f"nodes.npy saved — shape {nodes.shape}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_preprocessing()
