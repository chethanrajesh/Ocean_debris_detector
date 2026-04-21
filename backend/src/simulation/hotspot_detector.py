"""
hotspot_detector.py
Classifies ocean hotspots from simulation data.

Priority
--------
1. If trajectories.npy exists → derive hotspots from Lagrangian accumulation zones
   (same data source as the Trajectory view — ensures visual consistency)
2. Fallback → use future_predictions.npy (old GNN output)

Trajectory-based algorithm
--------------------------
1. Load trajectories.npy (N_particles, 14_snapshots, 5_features)
   features: [lat, lon, density, age_days, source_type]
2. Accumulate particle density on a 0.25° grid across ALL snapshots
   → cells with more particle visits = higher accumulation
3. Apply land mask
4. Threshold by accumulated density, classify critical/high/moderate
5. Attach movement vectors (from ocean_currents.npy) and source labels

This guarantees hotspot and trajectory views show the SAME physics.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# Snapshot schedule must match trajectory_simulator.py
SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90]

KNOWN_SOURCES_LABELED = [
    (30.0,  121.5, "Yangtze River Delta / East China Sea"),
    (23.0,  113.5, "Pearl River Delta, South China Sea"),
    (35.5,  139.8, "Pacific Coast Japan"),
    (10.0,  105.0, "Mekong Delta, Gulf of Thailand"),
    (23.7,   90.4, "Bay of Bengal / Bangladesh coast"),
    (13.0,   80.3, "Bay of Bengal / India SE coast"),
    ( 6.4,    3.4, "Gulf of Guinea / Nigeria coast"),
    (-6.2,  106.8, "Java Sea / Indonesia coast"),
    (18.5,  -69.9, "Caribbean Sea"),
    (-23.0, -43.2, "South Atlantic / Brazil coast"),
    (38.2,   15.6, "Mediterranean Sea / Italy"),
    (10.5,  -66.9, "Caribbean / Venezuela coast"),
    (32.0, -141.0, "North Pacific Garbage Patch"),
    (-32.0, -88.0, "South Pacific Gyre"),
    (28.0,  -63.0, "Sargasso Sea / North Atlantic Gyre"),
    (-26.0,  76.0, "Indian Ocean Gyre"),
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_ocean_mask() -> np.ndarray | None:
    p = DATA_DIR / "land_mask.npy"
    if not p.exists():
        logger.warning("land_mask.npy not found — land filtering disabled.")
        return None
    mask = np.load(str(p)).astype(bool)
    logger.info(f"Land mask loaded: shape {mask.shape}")
    return mask


def _ocean_mask_for_nodes(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n_lat, n_lon = mask.shape
    il = np.clip(
        np.round((nodes[:, 0] + 90  - RESOLUTION / 2) / RESOLUTION).astype(int),
        0, n_lat - 1,
    )
    jl = np.clip(
        np.round((nodes[:, 1] + 180 - RESOLUTION / 2) / RESOLUTION).astype(int),
        0, n_lon - 1,
    )
    return mask[il, jl]


def _load_currents_for_latlons(
    lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = DATA_DIR / "ocean_currents.npy"
    if not p.exists():
        return np.zeros(len(lats), np.float32), np.zeros(len(lats), np.float32)
    curr = np.load(str(p))
    n_lat, n_lon = curr.shape[:2]
    il = np.clip(
        np.round((lats + 90  - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lat - 1
    )
    jl = np.clip(
        np.round((lons + 180 - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lon - 1
    )
    return curr[il, jl, 0].astype(np.float32), curr[il, jl, 1].astype(np.float32)


def _nearest_source_label(lat: float, lon: float) -> str:
    best_label = "Open Ocean Accumulation Zone"
    best_dist  = float("inf")
    for slat, slon, label in KNOWN_SOURCES_LABELED:
        d = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5
        if d < best_dist:
            best_dist  = d
            best_label = label
    return best_label if best_dist <= 30 else "Open Ocean Accumulation Zone"


def _classify(density: float) -> str:
    if density > 0.7:   return "critical"
    if density > 0.4:   return "high"
    return "moderate"


def _trend(delta: float) -> str:
    if delta >  0.005:  return "increasing"
    if delta < -0.005:  return "decreasing"
    return "stable"


# ── Method 1: Trajectory-derived hotspots (preferred) ────────────────────────

def detect_from_trajectories(
    max_hotspots: int = 500,
) -> list[dict[str, Any]]:
    """
    Derive hotspots by accumulating particle density on a 0.25° grid
    across all 14 trajectory snapshots.

    A cell is a hotspot when multiple particles visit it — indicating
    a genuine accumulation zone, consistent with the Trajectory view.
    """
    traj_path = DATA_DIR / "trajectories.npy"
    if not traj_path.exists():
        return []

    traj = np.load(str(traj_path))          # (N_parts, N_snaps, 5)
    N_parts, N_snaps, _ = traj.shape
    logger.info(f"Building hotspots from trajectories: {N_parts} particles × {N_snaps} snapshots")

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)

    # Accumulation grid  [density_sum, particle_visit_count, latest_density]
    acc_density = np.zeros((n_lat, n_lon), dtype=np.float32)
    acc_count   = np.zeros((n_lat, n_lon), dtype=np.float32)
    # Track density change for trend (final vs day-0)
    density_day0  = np.zeros((n_lat, n_lon), dtype=np.float32)
    density_final = np.zeros((n_lat, n_lon), dtype=np.float32)

    # Weight later snapshots more (debris tends to accumulate over time)
    weights = np.linspace(0.5, 1.5, N_snaps)

    for snap_i in range(N_snaps):
        snap = traj[:, snap_i, :]           # (N_parts, 5)
        lats = snap[:, 0]
        lons = snap[:, 1]
        dens = snap[:, 2]
        src  = snap[:, 4]                  # 1=beached, skip in accumulation

        # Only count active + converging particles in the accumulation grid
        active = src != 1                  # exclude beached
        lats_a = lats[active]
        lons_a = lons[active]
        dens_a = dens[active]

        il = np.clip(
            np.round((lats_a + 90  - RESOLUTION / 2) / RESOLUTION).astype(int),
            0, n_lat - 1,
        )
        jl = np.clip(
            np.round((lons_a + 180 - RESOLUTION / 2) / RESOLUTION).astype(int),
            0, n_lon - 1,
        )
        w = weights[snap_i]
        np.add.at(acc_density, (il, jl), dens_a * w)
        np.add.at(acc_count,   (il, jl), w)

        if snap_i == 0:
            np.add.at(density_day0, (il, jl), dens_a)
        if snap_i == N_snaps - 1:
            np.add.at(density_final, (il, jl), dens_a)

    # Normalised mean density per cell
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_density = np.where(acc_count > 0, acc_density / acc_count, 0.0)

    if mean_density.max() > 0:
        mean_density /= mean_density.max()

    # Land-mask filter
    land_mask = _build_ocean_mask()
    if land_mask is not None:
        mean_density[~land_mask] = 0.0
        density_day0[~land_mask]  = 0.0
        density_final[~land_mask] = 0.0

    # Find occupied cells
    ri, ci = np.where(mean_density > 0.01)
    if len(ri) == 0:
        logger.warning("No accumulation cells above threshold — returning empty hotspots")
        return []

    cell_density = mean_density[ri, ci]

    # Cap to top max_hotspots
    if len(ri) > max_hotspots:
        top = np.argpartition(cell_density, -max_hotspots)[-max_hotspots:]
        ri, ci, cell_density = ri[top], ci[top], cell_density[top]

    # Convert grid indices back to lat/lon
    cell_lats = -89.875 + ri * RESOLUTION
    cell_lons = -179.875 + ci * RESOLUTION

    # Sample current vectors
    u_vec, v_vec = _load_currents_for_latlons(cell_lats, cell_lons)

    hotspots: list[dict[str, Any]] = []
    for k in range(len(ri)):
        lat   = float(cell_lats[k])
        lon   = float(cell_lons[k])
        d     = float(cell_density[k])
        d0    = float(density_day0[ri[k], ci[k]])
        df    = float(density_final[ri[k], ci[k]])
        delta = df - d0

        hotspots.append({
            "latitude":           round(lat, 4),
            "longitude":          round(lon, 4),
            "plastic_density":    round(d, 4),
            "level":              _classify(d),
            "accumulation_trend": _trend(delta),
            "movement_vector": {
                "u": round(float(u_vec[k]), 4),
                "v": round(float(v_vec[k]), 4),
            },
            "source_estimate":    _nearest_source_label(lat, lon),
        })

    hotspots.sort(key=lambda h: h["plastic_density"], reverse=True)
    logger.info(f"Detected {len(hotspots)} trajectory-derived hotspots")
    return hotspots


# ── Method 2: GNN prediction-based hotspots (fallback) ───────────────────────

def _load_predictions() -> tuple[np.ndarray, np.ndarray] | None:
    p = DATA_DIR / "future_predictions.npy"
    if not p.exists():
        return None
    preds = np.load(str(p))
    return preds[:, -1], preds[:, max(0, preds.shape[1] - 31)]


def _load_nodes() -> np.ndarray | None:
    p = DATA_DIR / "global_graph" / "nodes.npy"
    if not p.exists():
        return None
    return np.load(str(p))


def detect_from_predictions(max_hotspots: int = 500) -> list[dict[str, Any]]:
    """Fallback: GNN future_predictions.npy → hotspots."""
    result = _load_predictions()
    nodes  = _load_nodes()
    if result is None or nodes is None:
        return []

    final, prev = result
    N = min(len(final), len(nodes))
    final, prev, nodes = final[:N], prev[:N], nodes[:N]

    ocean_mask_arr = _build_ocean_mask()
    if ocean_mask_arr is not None:
        ocean_bool = _ocean_mask_for_nodes(nodes, ocean_mask_arr)
        final = final[ocean_bool]
        prev  = prev[ocean_bool]
        nodes = nodes[ocean_bool]

    if len(final) == 0:
        return []

    above = np.where(final > 0.005)[0]
    if len(above) > max_hotspots:
        top = np.argpartition(final[above], -max_hotspots)[-max_hotspots:]
        above = above[top]

    lats = nodes[above, 0]
    lons = nodes[above, 1]
    u_vec, v_vec = _load_currents_for_latlons(lats, lons)

    hotspots: list[dict[str, Any]] = []
    for i, idx in enumerate(above):
        lat = float(nodes[idx, 0])
        lon = float(nodes[idx, 1])
        d   = float(final[idx])
        hotspots.append({
            "latitude":           round(lat, 4),
            "longitude":          round(lon, 4),
            "plastic_density":    round(d, 4),
            "level":              _classify(d),
            "accumulation_trend": _trend(float(final[idx] - prev[idx])),
            "movement_vector": {
                "u": round(float(u_vec[i]), 4),
                "v": round(float(v_vec[i]), 4),
            },
            "source_estimate":    _nearest_source_label(lat, lon),
        })

    hotspots.sort(key=lambda h: h["plastic_density"], reverse=True)
    logger.info(f"Detected {len(hotspots)} GNN-based hotspots (fallback)")
    return hotspots


# ── Unified entry point ───────────────────────────────────────────────────────

def detect(max_hotspots: int = 500) -> list[dict[str, Any]]:
    """
    Unified hotspot detection.
    Prefers trajectory-derived hotspots (consistent with Trajectory view).
    Falls back to GNN predictions if trajectories not available.
    """
    traj_path = DATA_DIR / "trajectories.npy"
    if traj_path.exists():
        logger.info("Using trajectory-derived hotspot detection (consistent with Trajectory view)")
        result = detect_from_trajectories(max_hotspots)
        if result:
            return result
        logger.warning("Trajectory hotspot detection returned empty — falling back to GNN predictions")

    logger.info("Using GNN prediction-based hotspot detection (fallback)")
    return detect_from_predictions(max_hotspots)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    spots = detect()
    print(f"Total hotspots: {len(spots)}")
    for h in spots[:5]:
        print(h)
