"""
trajectory_simulator.py
90-day Lagrangian particle trajectory simulator.

Initialises particles from NOAA AOML / CYGNSS debris seed nodes and
advects them forward in time using:
  1. Blended CMEMS + OSCAR surface currents
  2. Stokes drift (wind-driven)
  3. Turbulent diffusion (random walk)

Particle state vector (5 features per particle):
  [lat, lon, mass_density, age_days, source_type]
    source_type: 0 = active (ocean), 1 = beached, 2 = converging

Snapshots are saved every 7 days → 14 snapshots (days 0, 7, 14, …, 90).

Output
------
data/trajectories.npy    : (N_particles, 14, 5)  float32
data/beaching_risk.npy   : (720, 1440)            float32  — cumulative beaching pressure
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ── Physics constants ─────────────────────────────────────────────────────────
EARTH_RADIUS_M      = 6.371e6
DT_SECONDS          = 86_400.0          # 1 day in seconds
STOKES_ALPHA        = 0.013             # Stokes drift coefficient (fraction of wind speed)
WINDAGE_BETA        = 0.01             # Direct wind drag (1%)
DIFFUSION_SIGMA_DEG = 0.15              # Turbulent diffusion std-dev per day (degrees) — increased for realism
DECAY_RATE          = 0.998             # Fragmentation: density multiplied each day
CONVERGENCE_MS      = 0.005            # Speed below which a particle is "converging" (very low — keep particles active)

# ── Snapshot schedule (days) ──────────────────────────────────────────────────
SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90]
N_SNAPSHOTS   = len(SNAPSHOT_DAYS)     # 14

# Source-type encoding
SRC_ACTIVE    = 0.0
SRC_BEACHED   = 1.0
SRC_CONVERGE  = 2.0


class TrajectorySimulator:
    """
    Physics-only (no GNN) 90-day Lagrangian particle simulator.

    Parameters
    ----------
    land_mask  : (720, 1440) bool   — True = ocean cell
    currents   : (720, 1440, 2) f32 — [u, v] m/s (blended CMEMS+OSCAR)
    winds      : (720, 1440, 2) f32 — [u10, v10] m/s (GFS)
    lats_grid  : (720,)  float32
    lons_grid  : (1440,) float32
    rng_seed   : int — for reproducible diffusion noise
    """

    def __init__(
        self,
        land_mask:  np.ndarray,
        currents:   np.ndarray,
        winds:      np.ndarray,
        lats_grid:  np.ndarray,
        lons_grid:  np.ndarray,
        rng_seed:   int = 42,
    ):
        self.land_mask = land_mask.astype(bool)
        self.currents  = currents.astype(np.float32)
        self.winds     = winds.astype(np.float32)
        self.lats_grid = lats_grid.astype(np.float32)
        self.lons_grid = lons_grid.astype(np.float32)
        self.rng       = np.random.default_rng(rng_seed)

        n_lat, n_lon = land_mask.shape
        self.n_lat = n_lat
        self.n_lon = n_lon

    # ── Grid lookup helpers ───────────────────────────────────────────────────

    def _lat_to_idx(self, lats: np.ndarray) -> np.ndarray:
        """Vectorised: lat array → grid row index (clamped)."""
        idx = np.round(
            (lats + 90 - RESOLUTION / 2) / RESOLUTION
        ).astype(int)
        return np.clip(idx, 0, self.n_lat - 1)

    def _lon_to_idx(self, lons: np.ndarray) -> np.ndarray:
        """Vectorised: lon array → grid col index (wrapped)."""
        idx = np.round(
            (lons + 180 - RESOLUTION / 2) / RESOLUTION
        ).astype(int)
        return idx % self.n_lon

    def _sample_field(
        self, field: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Nearest-neighbour sample of a (n_lat, n_lon, C) or (n_lat, n_lon) field."""
        il = self._lat_to_idx(lats)
        jl = self._lon_to_idx(lons)
        return field[il, jl]

    def _is_ocean(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Returns bool array: True where (lat, lon) is an ocean cell."""
        il = self._lat_to_idx(lats)
        jl = self._lon_to_idx(lons)
        return self.land_mask[il, jl]

    # ── Single-day step ───────────────────────────────────────────────────────

    def _step(
        self,
        lats:      np.ndarray,  # (N,) current latitudes
        lons:      np.ndarray,  # (N,) current longitudes
        density:   np.ndarray,  # (N,) mass density
        src_type:  np.ndarray,  # (N,) source-type encoding
        active_mask: np.ndarray, # (N,) bool — only move active particles
    ):
        """
        Advance all ACTIVE particles by one daily timestep.
        Returns (new_lats, new_lons, new_density, new_src_type).
        """
        N = len(lats)
        new_lats    = lats.copy()
        new_lons    = lons.copy()
        new_density = density.copy()
        new_src     = src_type.copy()

        if active_mask.sum() == 0:
            return new_lats, new_lons, new_density * DECAY_RATE, new_src

        # ── Sample ocean fields at particle positions ─────────────────────────
        curr_uv  = self._sample_field(self.currents, lats, lons)   # (N, 2)
        wind_uv  = self._sample_field(self.winds,    lats, lons)   # (N, 2)

        u_curr = curr_uv[:, 0]
        v_curr = curr_uv[:, 1]
        u_wind = wind_uv[:, 0]
        v_wind = wind_uv[:, 1]

        wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
        # Stokes drift: magnitude = STOKES_ALPHA × |wind|, direction from wind
        u_stokes = STOKES_ALPHA * u_wind
        v_stokes = STOKES_ALPHA * v_wind

        # Effective velocity (m/s)
        u_eff = u_curr + u_stokes + WINDAGE_BETA * u_wind
        v_eff = v_curr + v_stokes + WINDAGE_BETA * v_wind

        # Convergence detection
        speed = np.sqrt(u_curr ** 2 + v_curr ** 2)
        conv_mask = (speed < CONVERGENCE_MS) & active_mask
        new_src[conv_mask] = SRC_CONVERGE

        # ── Displacement (m/s → degrees) ─────────────────────────────────────
        cos_lat = np.cos(np.radians(np.clip(lats, -89.9, 89.9))) + 1e-9
        dlat = (v_eff * DT_SECONDS / EARTH_RADIUS_M) * (180.0 / np.pi)
        dlon = (u_eff * DT_SECONDS / (EARTH_RADIUS_M * cos_lat)) * (180.0 / np.pi)

        # Don't move converging or beached particles
        move_mask = active_mask & (src_type != SRC_BEACHED) & (src_type != SRC_CONVERGE)
        dlat[~move_mask] = 0.0
        dlon[~move_mask] = 0.0

        # ── Turbulent diffusion (only active moving particles) ────────────────
        noise_lat = self.rng.normal(0, DIFFUSION_SIGMA_DEG, N)
        noise_lon = self.rng.normal(0, DIFFUSION_SIGMA_DEG, N)
        dlat[move_mask] += noise_lat[move_mask]
        dlon[move_mask] += noise_lon[move_mask]

        # ── Project new positions ─────────────────────────────────────────────
        proj_lats = np.clip(lats + dlat, -89.9, 89.9)
        proj_lons = ((lons + dlon + 180) % 360) - 180

        # ── Beaching check ────────────────────────────────────────────────────
        ocean = self._is_ocean(proj_lats, proj_lons)
        beach_now = move_mask & ~ocean
        new_src[beach_now] = SRC_BEACHED

        # Apply positions only to particles that successfully moved to ocean
        moved = move_mask & ocean
        new_lats[moved] = proj_lats[moved]
        new_lons[moved] = proj_lons[moved]

        # ── Fragmentation decay ───────────────────────────────────────────────
        new_density = new_density * DECAY_RATE

        return new_lats, new_lons, new_density, new_src

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(
        self,
        seed_nodes: np.ndarray,   # (N, 3): [lat, lon, concentration_norm]
        n_days:     int = 90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run 90-day simulation.

        Parameters
        ----------
        seed_nodes : (N, 3) float32 — [lat, lon, concentration_norm]
        n_days     : int — simulation duration (default 90)

        Returns
        -------
        trajectories  : (N, N_SNAPSHOTS, 5) float32
        beaching_risk : (720, 1440) float32
        """
        seeds = seed_nodes.astype(np.float32)

        # Filter out seed nodes that start on land
        ocean_start = self._is_ocean(seeds[:, 0], seeds[:, 1])
        seeds = seeds[ocean_start]
        N = len(seeds)
        logger.info(f"Initialised {N:,} particles ({(~ocean_start).sum()} land-start filtered)")

        # Particle state arrays
        lats     = seeds[:, 0].copy()
        lons     = seeds[:, 1].copy()
        density  = seeds[:, 2].copy()
        age      = np.zeros(N, dtype=np.float32)
        src_type = np.full(N, SRC_ACTIVE, dtype=np.float32)

        # Output array
        snapshots = np.zeros((N, N_SNAPSHOTS, 5), dtype=np.float32)
        snap_idx  = 0

        # Beaching risk accumulator (how many particles hit each coastal cell)
        beach_risk = np.zeros((self.n_lat, self.n_lon), dtype=np.float32)

        # ── Day 0 snapshot ────────────────────────────────────────────────────
        snapshots[:, 0, 0] = lats
        snapshots[:, 0, 1] = lons
        snapshots[:, 0, 2] = density
        snapshots[:, 0, 3] = age
        snapshots[:, 0, 4] = src_type
        snap_idx = 1

        logger.info(f"Running {n_days}-day simulation, saving every 7 days...")

        for day in range(1, n_days + 1):
            active = src_type != SRC_BEACHED  # converging particles still get updated
            lats, lons, density, src_type = self._step(
                lats, lons, density, src_type, active
            )
            age += 1.0

            # Accumulate beaching risk for newly beached particles
            newly_beached = (src_type == SRC_BEACHED) & active
            if newly_beached.any():
                bi = self._lat_to_idx(lats[newly_beached])
                bj = self._lon_to_idx(lons[newly_beached])
                np.add.at(beach_risk, (bi, bj), density[newly_beached])

            # Save snapshot at each 7-day interval
            if day in SNAPSHOT_DAYS and snap_idx < N_SNAPSHOTS:
                snapshots[:, snap_idx, 0] = lats
                snapshots[:, snap_idx, 1] = lons
                snapshots[:, snap_idx, 2] = density
                snapshots[:, snap_idx, 3] = age
                snapshots[:, snap_idx, 4] = src_type
                snap_idx += 1

                n_active  = int((src_type == SRC_ACTIVE).sum())
                n_beached = int((src_type == SRC_BEACHED).sum())
                n_conv    = int((src_type == SRC_CONVERGE).sum())
                logger.info(
                    f"  Day {day:3d}/{n_days} | active={n_active:,} "
                    f"beached={n_beached:,} converging={n_conv:,} "
                    f"mean_density={density.mean():.4f}"
                )

        # Normalise beaching risk to [0, 1]
        if beach_risk.max() > 0:
            beach_risk /= beach_risk.max()
        beach_risk[~self.land_mask] = 0.0   # only coastal cells matter

        return snapshots, beach_risk


# ── Convenience runner ────────────────────────────────────────────────────────

def run_trajectory_simulation(
    n_days: int = 90,
    max_particles: int = 5000,
    force_reseed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all required data, run simulation, save outputs.

    Returns
    -------
    trajectories  : (N, N_SNAPSHOTS, 5) float32
    beaching_risk : (720, 1440) float32
    """
    from src.data_pipeline.fetch_aoml import load_aoml_seeds

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_grid = np.linspace(-90 + RESOLUTION / 2,  90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    def _load(fname, shape, is_bool=False):
        p = DATA_DIR / fname
        if p.exists():
            arr = np.load(str(p))
            logger.info(f"Loaded {fname} — shape {arr.shape}")
            return arr
        logger.warning(f"{fname} not found — using zeros {shape}")
        return np.zeros(shape, dtype=bool if is_bool else np.float32)

    land_mask = _load("land_mask.npy",       (n_lat, n_lon),    is_bool=True).astype(bool)
    currents  = _load("ocean_currents.npy",  (n_lat, n_lon, 2))
    winds     = _load("wind_data.npy",       (n_lat, n_lon, 2))

    if np.abs(currents).max() < 1e-6:
        raise RuntimeError(
            "ocean_currents.npy is all-zero. Run the data pipeline first: "
            "python -m src.data_pipeline.fetch_cmems"
        )
    if np.abs(winds).max() < 1e-6:
        raise RuntimeError(
            "wind_data.npy is all-zero. Run the data pipeline first: "
            "python -m src.data_pipeline.fetch_noaa"
        )

    # ── Load seed nodes ───────────────────────────────────────────────────────
    seeds = load_aoml_seeds()
    logger.info(f"AOML seeds: {len(seeds):,} nodes before filtering")

    # Deduplicate to 0.25° grid resolution
    seen = set()
    dedup = []
    for row in seeds:
        key = (round(float(row[0]), 2), round(float(row[1]), 2))
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    seeds = np.array(dedup, dtype=np.float32)
    logger.info(f"After dedup: {len(seeds):,} unique grid cells")

    # Cap particle count for performance
    if len(seeds) > max_particles:
        top_idx = np.argpartition(seeds[:, 2], -max_particles)[-max_particles:]
        seeds = seeds[top_idx]
        logger.info(f"Capped to top {max_particles:,} seeds by concentration")

    # ── Run simulation ────────────────────────────────────────────────────────
    sim = TrajectorySimulator(land_mask, currents, winds, lats_grid, lons_grid)
    trajectories, beaching_risk = sim.run(seeds, n_days=n_days)

    # ── Save outputs ──────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traj_path  = DATA_DIR / "trajectories.npy"
    beach_path = DATA_DIR / "beaching_risk.npy"

    np.save(str(traj_path),  trajectories)
    np.save(str(beach_path), beaching_risk)
    logger.info(f"Saved trajectories.npy   — shape {trajectories.shape}")
    logger.info(f"Saved beaching_risk.npy  — shape {beaching_risk.shape}")

    return trajectories, beaching_risk


if __name__ == "__main__":
    import argparse
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run 90-day Lagrangian trajectory simulation")
    parser.add_argument("--days",          type=int, default=90,   help="Simulation duration in days")
    parser.add_argument("--max-particles", type=int, default=5000, help="Max number of particles")
    args = parser.parse_args()
    traj, risk = run_trajectory_simulation(n_days=args.days, max_particles=args.max_particles)
    print(f"trajectories.npy shape:  {traj.shape}")
    print(f"beaching_risk.npy shape: {risk.shape}")
    print(f"Max beaching risk: {risk.max():.4f}")
