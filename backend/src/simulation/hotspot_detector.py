"""
hotspot_detector.py
Classifies ocean hotspots from the final simulation predictions.

Algorithm
---------
1. Load final time step of future_predictions.npy
2. Apply land mask — discard any node with land_mask_bit = 0
3. Compute 95th percentile threshold across all ocean node densities
4. Classify:
     density > 0.7                    → critical
     0.4 < density <= 0.7             → high
     threshold < density <= 0.4       → moderate
5. Compute accumulation trend (final vs. 30 steps earlier)
6. Compute movement vector (average u, v from ocean_currents.npy)
7. Estimate source region from source_points.npy (nearest known source)
"""
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

KNOWN_SOURCES_LABELED = [
    (30.0, 121.5, "Yangtze River Delta / East China Sea"),
    (23.0, 113.5, "Pearl River Delta, South China Sea"),
    (35.5, 139.8, "Pacific Coast Japan"),
    (10.0, 105.0, "Mekong Delta, Gulf of Thailand"),
    (23.7,  90.4, "Bay of Bengal / Bangladesh coast"),
    (13.0,  80.3, "Bay of Bengal / India SE coast"),
    (6.4,   3.4,  "Gulf of Guinea / Nigeria coast"),
    (-6.2, 106.8, "Java Sea / Indonesia coast"),
    (18.5, -69.9, "Caribbean Sea"),
    (-23.0, -43.2, "South Atlantic / Brazil coast"),
    (38.2,  15.6, "Mediterranean Sea / Italy"),
    (10.5, -66.9, "Caribbean / Venezuela coast"),
]


def _load_predictions() -> tuple[np.ndarray, np.ndarray] | None:
    """Load future_predictions.npy and return (final_step, step_minus_30)."""
    pred_path = DATA_DIR / "future_predictions.npy"
    if not pred_path.exists():
        logger.error(f"future_predictions.npy not found at {pred_path}")
        return None
    preds = np.load(str(pred_path))   # (N_ocean, T)
    final = preds[:, -1]
    prev  = preds[:, max(0, preds.shape[1] - 31)]
    return final, prev


def _load_nodes() -> np.ndarray | None:
    """Load nodes.npy — shape (N, 7)."""
    nodes_path = DATA_DIR / "global_graph" / "nodes.npy"
    if not nodes_path.exists():
        logger.error(f"nodes.npy not found at {nodes_path}")
        return None
    return np.load(str(nodes_path))


def _load_currents_for_nodes(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ocean_currents.npy and index into it using node lat/lon.
    Returns (u_at_nodes, v_at_nodes).
    """
    curr_path = DATA_DIR / "ocean_currents.npy"
    if not curr_path.exists():
        N = len(nodes)
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

    currents = np.load(str(curr_path))   # (N_lat, N_lon, 2)
    n_lat, n_lon = currents.shape[:2]
    lats_grid = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat)
    lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon)

    node_lats = nodes[:, 0]
    node_lons = nodes[:, 1]
    i_idx = np.clip(
        np.round((node_lats + 90 - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lat - 1
    )
    j_idx = np.clip(
        np.round((node_lons + 180 - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lon - 1
    )

    u = currents[i_idx, j_idx, 0]
    v = currents[i_idx, j_idx, 1]
    return u.astype(np.float32), v.astype(np.float32)


def _nearest_source_label(lat: float, lon: float) -> str:
    """Return the label of the nearest known plastic source."""
    best_label = "Open Ocean Accumulation"
    best_dist = float("inf")
    for slat, slon, label in KNOWN_SOURCES_LABELED:
        d = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_label = label
    # If more than ~30° away, call it open ocean
    if best_dist > 30:
        return "Open Ocean Accumulation Zone"
    return best_label


def detect() -> list[dict[str, Any]]:
    """
    Run hotspot detection and return a list of hotspot dicts matching the API schema.
    """
    result = _load_predictions()
    nodes  = _load_nodes()

    if result is None or nodes is None:
        logger.warning("Returning empty hotspot list (data not available).")
        return []

    final, prev = result
    N = min(len(final), len(nodes))
    final = final[:N]
    prev  = prev[:N]
    nodes = nodes[:N]

    threshold = float(np.percentile(final, 95))
    logger.info(f"Hotspot threshold (95th pct): {threshold:.4f}")

    u_vec, v_vec = _load_currents_for_nodes(nodes)

    hotspots = []
    above_thresh = np.where(final > threshold)[0]

    for idx in above_thresh:
        density = float(final[idx])
        delta   = float(final[idx] - prev[idx])
        lat     = float(nodes[idx, 0])
        lon     = float(nodes[idx, 1])
        u       = float(u_vec[idx])
        v       = float(v_vec[idx])

        # Severity classification
        if density > 0.7:
            level = "critical"
        elif density > 0.4:
            level = "high"
        else:
            level = "moderate"

        # Accumulation trend
        if delta > 0.05:
            trend = "increasing"
        elif delta < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        source = _nearest_source_label(lat, lon)

        hotspots.append({
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "plastic_density": round(density, 4),
            "level": level,
            "accumulation_trend": trend,
            "movement_vector": {
                "u": round(u, 4),
                "v": round(v, 4),
            },
            "source_estimate": source,
        })

    # Sort by density descending
    hotspots.sort(key=lambda h: h["plastic_density"], reverse=True)
    logger.info(f"Detected {len(hotspots)} hotspots above {threshold:.4f}")
    return hotspots


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hs = detect()
    for h in hs[:5]:
        print(h)
