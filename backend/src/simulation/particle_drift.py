"""
particle_drift.py
Lagrangian particle-drift physics engine.

For each 6-hour time step, computes displacement as:
  displacement = (α × current_vector) + (β × wind_drag_vector) + (γ × stokes_drift_vector)

  α = 1.0  — full current advection
  β = 0.03 — 3% windage (standard drifter value)
  γ = 1.0  — Stokes drift

Coastline collision:
  When a projected position falls on a land node, decompose displacement into
  normal and tangential components relative to the land boundary.
  Retain only the tangential component (debris slides along the coast).

Convergence detection:
  Nodes where |current| < 0.05 m/s are flagged as convergence candidates.
  Density at convergence nodes accumulates without positional advance.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Physics coefficients
ALPHA = 1.0    # current advection weight
BETA  = 0.03   # wind drag coefficient (3% windage)
GAMMA = 1.0    # Stokes drift weight

DT_HOURS = 6.0                   # time step in hours
DT_SECONDS = DT_HOURS * 3600.0  # 21600 s

CONVERGENCE_THRESHOLD = 0.05  # m/s — below this → convergence zone
EARTH_RADIUS_M = 6.371e6      # metres


def latlon_displacement(
    lats: np.ndarray, lons: np.ndarray,
    du_ms: np.ndarray, dv_ms: np.ndarray,
    dt: float = DT_SECONDS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert velocity (m/s) to displacement in degrees of lat/lon over dt seconds.

    Parameters
    ----------
    lats  : (N,) current latitudes in degrees
    lons  : (N,) current longitudes in degrees
    du_ms : (N,) eastward velocity (m/s)
    dv_ms : (N,) northward velocity (m/s)
    dt    : time step in seconds

    Returns
    -------
    dlat : (N,) latitude change (degrees)
    dlon : (N,) longitude change (degrees)
    """
    dlat = (dv_ms * dt / EARTH_RADIUS_M) * (180.0 / np.pi)
    dlon = (du_ms * dt / (EARTH_RADIUS_M * np.cos(np.radians(lats)) + 1e-9)) * (180.0 / np.pi)
    return dlat, dlon


def _nearest_ocean_node(
    lat: float, lon: float,
    lats_grid: np.ndarray, lons_grid: np.ndarray,
    node_id: np.ndarray
) -> tuple[int, int]:
    """Find the nearest ocean-node grid cell to a projected position."""
    i = int(np.argmin(np.abs(lats_grid - lat)))
    j = int(np.argmin(np.abs(lons_grid - lon))) % len(lons_grid)
    if node_id[i, j] >= 0:
        return i, j
    # Spiral search
    for radius in range(1, 20):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni = i + di
                nj = (j + dj) % len(lons_grid)
                if 0 <= ni < len(lats_grid) and node_id[ni, nj] >= 0:
                    return ni, nj
    return i, j  # fallback


def _tangential_displacement(
    dlat: float, dlon: float,
    lat_i: float, lon_i: float,
    lat_land: float, lon_land: float
) -> tuple[float, float]:
    """
    Given a displacement (dlat, dlon) that would move into a land cell,
    decompose it into normal (toward land) and tangential components.
    Returns the tangential-only displacement.
    """
    # Normal vector: from ocean cell toward land cell
    n_lat = lat_land - lat_i
    n_lon = lon_land - lon_i
    n_mag = np.sqrt(n_lat**2 + n_lon**2) + 1e-12
    n_lat /= n_mag
    n_lon /= n_mag

    # Dot product (normal component)
    dot = dlat * n_lat + dlon * n_lon
    # Tangential = total - normal component
    t_dlat = dlat - dot * n_lat
    t_dlon = dlon - dot * n_lon
    return t_dlat, t_dlon


class LagrangianEngine:
    """
    Lagrangian particle drift engine over a pre-built ocean node graph.

    Parameters
    ----------
    lats_grid : (N_lat,) 1-D array of grid latitudes
    lons_grid : (N_lon,) 1-D array of grid longitudes
    land_mask : (N_lat, N_lon) boolean — True = ocean
    node_id   : (N_lat, N_lon) int32 — ocean node index (-1 for land)
    """

    def __init__(
        self,
        lats_grid: np.ndarray,
        lons_grid: np.ndarray,
        land_mask: np.ndarray,
        node_id: np.ndarray,
    ):
        self.lats_grid = lats_grid
        self.lons_grid = lons_grid
        self.land_mask = land_mask
        self.node_id   = node_id
        self.resolution = float(lats_grid[1] - lats_grid[0])

    def step(
        self,
        node_lats: np.ndarray,     # (N_ocean,)
        node_lons: np.ndarray,     # (N_ocean,)
        density:   np.ndarray,     # (N_ocean,) current plastic density
        u_curr:    np.ndarray,     # (N_ocean,) m/s
        v_curr:    np.ndarray,     # (N_ocean,) m/s
        u_wind:    np.ndarray,     # (N_ocean,) m/s
        v_wind:    np.ndarray,     # (N_ocean,) m/s
        stokes:    np.ndarray,     # (N_ocean,) m/s magnitude (E-ward approx)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance all ocean nodes by one 6-hour time step.

        Returns
        -------
        new_density      : (N_ocean,) updated density after drift
        convergence_flag : (N_ocean,) boolean — True if node is convergence zone
        """
        N = len(node_lats)

        # ── Compute effective velocity ──────────────────────────────────────
        # Wind drag decomposes to approximate Stokes direction (simplification)
        u_stokes = stokes * np.sign(u_wind + 1e-12)
        v_stokes = stokes * np.sign(v_wind + 1e-12) * 0.5

        u_eff = ALPHA * u_curr + BETA * u_wind + GAMMA * u_stokes
        v_eff = ALPHA * v_curr + BETA * v_wind + GAMMA * v_stokes

        # ── Convergence detection ───────────────────────────────────────────
        speed = np.sqrt(u_curr**2 + v_curr**2)
        convergence_flag = speed < CONVERGENCE_THRESHOLD

        # ── Compute displacements ───────────────────────────────────────────
        dlat, dlon = latlon_displacement(node_lats, node_lons, u_eff, v_eff)

        # Don't move convergence-zone particles
        dlat[convergence_flag] = 0.0
        dlon[convergence_flag] = 0.0

        # ── Project new positions ────────────────────────────────────────────
        new_lats = np.clip(node_lats + dlat, -89.9, 89.9)
        new_lons = ((node_lons + dlon + 180) % 360) - 180

        # ── Build output density array (redistribute mass) ──────────────────
        n_lat = len(self.lats_grid)
        n_lon = len(self.lons_grid)
        N_ocean = len(node_lats)
        new_density_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

        for idx in range(N):
            if convergence_flag[idx]:
                # Accumulate in place
                i_src = int(np.argmin(np.abs(self.lats_grid - node_lats[idx])))
                j_src = int(np.argmin(np.abs(self.lons_grid - node_lons[idx])))
                new_density_grid[i_src, j_src] += density[idx] * 1.02  # grow 2%
                continue

            lat_p = new_lats[idx]
            lon_p = new_lons[idx]

            i_dst = int(np.argmin(np.abs(self.lats_grid - lat_p)))
            j_dst = int(np.argmin(np.abs(self.lons_grid - lon_p)))

            if not self.land_mask[i_dst, j_dst]:
                # Hit land — apply tangential deflection
                i_src = int(np.argmin(np.abs(self.lats_grid - node_lats[idx])))
                j_src = int(np.argmin(np.abs(self.lons_grid - node_lons[idx])))
                td_lat, td_lon = _tangential_displacement(
                    dlat[idx], dlon[idx],
                    node_lats[idx], node_lons[idx],
                    self.lats_grid[i_dst], self.lons_grid[j_dst]
                )
                lat_p2 = np.clip(node_lats[idx] + td_lat, -89.9, 89.9)
                lon_p2 = ((node_lons[idx] + td_lon + 180) % 360) - 180
                i_dst = int(np.argmin(np.abs(self.lats_grid - lat_p2)))
                j_dst = int(np.argmin(np.abs(self.lons_grid - lon_p2)))
                # If still on land, deposit at source
                if not self.land_mask[i_dst, j_dst]:
                    i_dst, j_dst = i_src, j_src

            new_density_grid[i_dst, j_dst] += density[idx]

        # Apply land mask
        new_density_grid[~self.land_mask] = 0.0

        # Extract ocean-node values in order
        ocean_rows, ocean_cols = np.where(self.land_mask)
        new_density = new_density_grid[ocean_rows, ocean_cols].astype(np.float32)

        # Normalize to [0, 1] per step
        mx = new_density.max()
        if mx > 1e-8:
            new_density /= mx

        return new_density, convergence_flag
