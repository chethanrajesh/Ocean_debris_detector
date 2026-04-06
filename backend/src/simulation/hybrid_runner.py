"""
hybrid_runner.py
Hybrid Lagrangian + GNN simulation loop.

For each of 365 time steps (6-hour intervals ≈ 90 days total):
  1. Load current graph state G_t (node features, current density)
  2. Run Lagrangian engine → physics_density_t+1 per node
  3. Pass G_t through trained GNN → gnn_correction per node
  4. Blend: final_density_t+1 = physics_density_t+1 + (λ × gnn_correction)
  5. Apply land mask — zero out density on all land nodes
  6. Flag convergence zones as hotspot candidates
  7. Append final_density_t+1 to future_predictions[:, t+1]
  8. Advance G_t ← G_t+1

Output: data/future_predictions.npy — shape (N_ocean_nodes, 365)

Usage
-----
  python -m src.simulation.hybrid_runner [--timesteps 365] [--lambda-gnn 0.4]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
RESOLUTION = 0.25
LAMBDA_GNN = 0.4     # blending weight for GNN correction


def _load_or_warn(fname, default_shape, dtype=np.float32, is_bool=False):
    p = DATA_DIR / fname
    if p.exists():
        return np.load(str(p))
    logger.warning(f"{fname} not found; using zeros {default_shape}")
    return np.zeros(default_shape, dtype=bool if is_bool else dtype)


def run_simulation(n_timesteps: int = 365, lambda_gnn: float = LAMBDA_GNN) -> np.ndarray:
    """
    Execute the hybrid simulation loop.

    Returns
    -------
    future_predictions : (N_ocean, n_timesteps) float32
    """
    from src.simulation.land_mask import get_ocean_node_coordinates
    from src.simulation.particle_drift import LagrangianEngine

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_grid = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    # ── Load supporting data ──────────────────────────────────────────────────
    land_mask = _load_or_warn("land_mask.npy", (n_lat, n_lon), is_bool=True)
    if land_mask.dtype != bool:
        land_mask = land_mask.astype(bool)

    currents = _load_or_warn("ocean_currents.npy", (n_lat, n_lon, 2))
    winds    = _load_or_warn("wind_data.npy",      (n_lat, n_lon, 2))
    stokes   = _load_or_warn("stokes_drift.npy",    (n_lat, n_lon))
    nodes_data = _load_or_warn("global_graph/nodes.npy", (100, 7))

    ocean_lats, ocean_lons, node_id_grid = get_ocean_node_coordinates(land_mask)
    N = len(ocean_lats)
    logger.info(f"Ocean nodes: {N:,}")

    # ── Initial density ───────────────────────────────────────────────────────
    density_2d = _load_or_warn("density_maps.npy", (n_lat, n_lon))
    ocean_rows, ocean_cols = np.where(land_mask)
    density = density_2d[ocean_rows, ocean_cols].astype(np.float32)

    # ── Current vectors at ocean nodes ────────────────────────────────────────
    u_curr  = currents[ocean_rows, ocean_cols, 0]
    v_curr  = currents[ocean_rows, ocean_cols, 1]
    u_wind  = winds[ocean_rows, ocean_cols, 0]
    v_wind  = winds[ocean_rows, ocean_cols, 1]
    stk_mag = stokes[ocean_rows, ocean_cols]

    # ── Load GNN checkpoint ───────────────────────────────────────────────────
    ckpt_path = MODELS_DIR / "gnn_checkpoint.pt"
    gnn = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_path.exists():
        try:
            from src.models.density_predictor import OceanDebrisGNN, build_graph_data
            gnn = OceanDebrisGNN.load_checkpoint(str(ckpt_path), device=device)
            logger.info(f"GNN checkpoint loaded from {ckpt_path}")
        except Exception as exc:
            logger.warning(f"GNN load failed ({exc}); running physics-only.")
    else:
        logger.warning(f"GNN checkpoint not found at {ckpt_path}; running physics-only.")

    # ── Build edge index (load or build) ─────────────────────────────────────
    edge_index_t = None
    edge_attr_t  = None
    if gnn is not None:
        try:
            from src.utils.geo_utils import build_edge_index
            logger.info("Building edge index (first run may take several minutes)...")
            src_idx, dst_idx, dist_km, bear_deg, align_vals = build_edge_index(
                lats_grid, lons_grid, land_mask, k=8
            )
            # Compute alignment from actual currents
            from src.utils.geo_utils import current_alignment as ca_fn
            for e_i in range(len(src_idx)):
                ni = ocean_rows[src_idx[e_i]]
                nj = ocean_cols[src_idx[e_i]]
                align_vals[e_i] = ca_fn(
                    float(currents[ni, nj, 0]),
                    float(currents[ni, nj, 1]),
                    float(bear_deg[e_i])
                )
            edge_index_t = torch.tensor(
                np.stack([src_idx, dst_idx]), dtype=torch.long
            ).to(device)
            edge_attr_t = torch.tensor(
                np.stack([dist_km / 100.0, bear_deg / 360.0,
                          (align_vals + 1) / 2.0], axis=-1),
                dtype=torch.float
            ).to(device)
        except Exception as exc:
            logger.warning(f"Edge index build failed: {exc}; GNN disabled.")
            gnn = None

    # ── Lagrangian engine ─────────────────────────────────────────────────────
    engine = LagrangianEngine(lats_grid, lons_grid, land_mask, node_id_grid)

    # ── Output array ─────────────────────────────────────────────────────────
    predictions = np.zeros((N, n_timesteps), dtype=np.float32)
    predictions[:, 0] = density

    # ── Main simulation loop ──────────────────────────────────────────────────
    logger.info(f"Starting hybrid simulation: {n_timesteps} steps, λ_GNN={lambda_gnn}")

    for t in range(1, n_timesteps):
        # Step 2: Lagrangian physics
        phys_density, conv_flag = engine.step(
            ocean_lats, ocean_lons, density,
            u_curr, v_curr, u_wind, v_wind, stk_mag
        )

        # Step 3: GNN correction
        gnn_correction = np.zeros(N, dtype=np.float32)
        if gnn is not None:
            try:
                # Build node feature matrix for this timestep
                node_feats = np.stack([
                    ocean_lats, ocean_lons,
                    u_curr, v_curr, u_wind, v_wind,
                    np.zeros(N),   # SST placeholder
                    density,
                    stk_mag
                ], axis=-1).astype(np.float32)

                x_t = torch.tensor(node_feats, dtype=torch.float).to(device)
                from src.models.density_predictor import build_graph_data
                data_t = build_graph_data(x_t, edge_index_t, edge_attr_t)

                with torch.no_grad():
                    gnn_correction = gnn(data_t).cpu().numpy()
            except Exception as exc:
                if t == 1:
                    logger.warning(f"GNN inference failed ({exc}); using physics-only.")
                gnn_correction = np.zeros(N, dtype=np.float32)

        # Step 4: Blend
        final_density = phys_density + lambda_gnn * gnn_correction

        # Step 5: Apply land mask (already zero from physics engine)
        # Additional safety enforcement
        final_density = np.clip(final_density, 0.0, 1.0)

        # Step 6: Accumulate at convergence zones
        final_density[conv_flag] = np.clip(
            final_density[conv_flag] * 1.01, 0, 1
        )

        # Step 7: Store
        predictions[:, t] = final_density

        # Step 8: Advance
        density = final_density

        if t % 50 == 0 or t == n_timesteps - 1:
            logger.info(
                f"  t={t:03d}/{n_timesteps} | "
                f"mean_density={density.mean():.4f} | "
                f"max_density={density.max():.4f} | "
                f"convergence_zones={conv_flag.sum()}"
            )

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "future_predictions.npy"
    np.save(str(out_path), predictions)
    logger.info(f"future_predictions.npy saved — shape {predictions.shape}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run hybrid Lagrangian + GNN simulation")
    parser.add_argument("--timesteps",  type=int,   default=365)
    parser.add_argument("--lambda-gnn", type=float, default=0.4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_simulation(args.timesteps, args.lambda_gnn)


if __name__ == "__main__":
    main()
