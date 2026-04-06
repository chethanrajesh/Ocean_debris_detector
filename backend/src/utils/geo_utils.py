"""
geo_utils.py
Geographic utility functions for the ocean debris system.
"""
import numpy as np
from typing import Tuple


EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance (km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the initial compass bearing (degrees) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def bearing_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized bearing in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def current_alignment(u: float, v: float, bearing_deg: float) -> float:
    """
    Dot product of current vector (u, v) with the unit direction vector
    corresponding to bearing_deg. Returns value in [-1, 1].
    """
    bearing_rad = np.radians(bearing_deg)
    dir_u = np.sin(bearing_rad)   # East component of bearing direction
    dir_v = np.cos(bearing_rad)   # North component
    speed = np.sqrt(u**2 + v**2)
    if speed < 1e-9:
        return 0.0
    return float((u * dir_u + v * dir_v) / speed)


def build_ocean_grid(resolution: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global lat/lon grid at the given resolution.
    Returns (lats, lons) arrays of shape (num_lats,) and (num_lons,).
    """
    lats = np.arange(-90 + resolution / 2, 90, resolution)
    lons = np.arange(-180 + resolution / 2, 180, resolution)
    return lats, lons


def build_edge_index(
    lats: np.ndarray,
    lons: np.ndarray,
    ocean_mask: np.ndarray,
    k: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a CSR-style edge list connecting each ocean node to up to k=8
    geographic neighbours (N, NE, E, SE, S, SW, W, NW).

    Parameters
    ----------
    lats : (N_lat,) array
    lons : (N_lon,) array
    ocean_mask : (N_lat, N_lon) boolean array — True = ocean
    k : number of neighbour directions (8 for full compass)

    Returns
    -------
    src_idx, dst_idx : COO edge indices (both (E,))
    dist_km         : edge distances in km  (E,)
    bear_deg        : bearing from src to dst (E,)
    align_vals      : placeholder alignment array (E,) — filled in hybrid_runner
    """
    n_lat, n_lon = len(lats), len(lons)

    # Build flat node index for ocean cells only
    node_id = np.full((n_lat, n_lon), -1, dtype=np.int32)
    ocean_indices = np.argwhere(ocean_mask)  # (N, 2)
    for idx, (i, j) in enumerate(ocean_indices):
        node_id[i, j] = idx

    # 8-directional offsets: (di, dj)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    src_list, dst_list, dist_list, bear_list = [], [], [], []

    for idx, (i, j) in enumerate(ocean_indices):
        lat1, lon1 = lats[i], lons[j]
        for di, dj in directions:
            ni = i + di
            nj = (j + dj) % n_lon   # wrap longitude
            if 0 <= ni < n_lat and ocean_mask[ni, nj]:
                dst = node_id[ni, nj]
                if dst < 0:
                    continue
                lat2, lon2 = lats[ni], lons[nj]
                d = haversine(lat1, lon1, lat2, lon2)
                b = bearing(lat1, lon1, lat2, lon2)
                src_list.append(idx)
                dst_list.append(dst)
                dist_list.append(d)
                bear_list.append(b)

    src_idx = np.array(src_list, dtype=np.int64)
    dst_idx = np.array(dst_list, dtype=np.int64)
    dist_km = np.array(dist_list, dtype=np.float32)
    bear_deg = np.array(bear_list, dtype=np.float32)
    align_vals = np.zeros(len(src_list), dtype=np.float32)

    return src_idx, dst_idx, dist_km, bear_deg, align_vals
