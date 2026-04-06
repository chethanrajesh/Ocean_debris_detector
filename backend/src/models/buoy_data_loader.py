"""
buoy_data_loader.py
Loads NOAA Global Drifter Program buoy trajectories and constructs
PyTorch Geometric Data objects (local graph patches) for GNN training.

Each training sample:
  - Centred on a buoy GPS observation at time t
  - Graph patch = the node containing the buoy + its 8 geographic neighbours
  - Node features x_i (dim 9):
      [lat, lon, u_current, v_current, u_wind, v_wind, sst,
       plastic_density_t, stokes_drift_magnitude]
  - Edge features e_ij (dim 3):
      [distance_km, bearing_deg, current_alignment_score]
  - Label y = actual buoy displacement at t+1 (used to train density correction)

Train/val/test split: 80/10/10 by trajectory ID (no data leakage).
"""
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25


try:
    from torch_geometric.data import Data, Dataset
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    # Stub classes
    class Dataset:
        pass
    class Data:
        pass


def _latlon_to_ij(lat: float, lon: float, n_lat: int, n_lon: int):
    i = int(round((lat + 90 - RESOLUTION / 2) / RESOLUTION))
    j = int(round((lon + 180 - RESOLUTION / 2) / RESOLUTION))
    return max(0, min(n_lat - 1, i)), max(0, min(n_lon - 1, j))


class BuoyGraphDataset(Dataset if TG_AVAILABLE else object):
    """
    PyG Dataset built from NOAA GDP buoy trajectories.

    Parameters
    ----------
    split     : 'train', 'val', or 'test'
    split_seed: random seed for reproducible trajectory ID split
    max_samples: max total samples to load (None = all)
    """

    def __init__(
        self,
        split: str = "train",
        split_seed: int = 42,
        max_samples: int = None,
    ):
        if not TG_AVAILABLE:
            raise ImportError("torch-geometric is required for GNN training.")

        super().__init__()
        self.split = split
        self._load_supporting_data()
        self._build_samples(split, split_seed, max_samples)

    # ── Supporting arrays from preprocessing ─────────────────────────────────

    def _load_supporting_data(self):
        n_lat = int(180 / RESOLUTION)
        n_lon = int(360 / RESOLUTION)
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.lats_grid = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat, dtype=np.float32)
        self.lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

        def _try_load(fname, default_shape, dtype=np.float32):
            p = DATA_DIR / fname
            if p.exists():
                return np.load(str(p))
            logger.warning(f"{fname} not found; using zeros {default_shape}")
            return np.zeros(default_shape, dtype=dtype)

        self.land_mask    = _try_load("land_mask.npy",   (n_lat, n_lon), dtype=bool)
        self.currents     = _try_load("ocean_currents.npy", (n_lat, n_lon, 2))
        self.winds        = _try_load("wind_data.npy",      (n_lat, n_lon, 2))
        self.density_map  = _try_load("density_maps.npy",   (n_lat, n_lon))
        self.stokes       = _try_load("stokes_drift.npy",   (n_lat, n_lon))

    # ── Build training samples from buoy trajectories ─────────────────────────

    def _build_samples(self, split: str, seed: int, max_samples):
        traj_path = DATA_DIR / "buoy_trajectories.pkl"
        if not traj_path.exists():
            logger.error("buoy_trajectories.pkl not found. Run fetch_noaa.py first.")
            self._samples = []
            return

        df = pd.read_pickle(str(traj_path))
        df = df.dropna(subset=["lat", "lon", "u_obs", "v_obs"])
        df = df.sort_values(["buoy_id", "timestamp"])

        all_buoy_ids = df["buoy_id"].unique().tolist()
        random.seed(seed)
        random.shuffle(all_buoy_ids)

        n = len(all_buoy_ids)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        if split == "train":
            ids = all_buoy_ids[:n_train]
        elif split == "val":
            ids = all_buoy_ids[n_train:n_train + n_val]
        else:
            ids = all_buoy_ids[n_train + n_val:]

        subset = df[df["buoy_id"].isin(ids)].reset_index(drop=True)
        logger.info(f"[{split}] {len(ids)} buoys, {len(subset)} records")

        # Build (t, t+1) consecutive pairs grouped by buoy
        samples = []
        for bid, group in subset.groupby("buoy_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            for k in range(len(group) - 1):
                row_t  = group.iloc[k]
                row_t1 = group.iloc[k + 1]
                # Only use 6-hourly steps (within 7-hr window)
                dt = (row_t1["timestamp"] - row_t["timestamp"]).total_seconds()
                if not (18000 <= dt <= 25200):
                    continue
                samples.append((row_t, row_t1))
                if max_samples and len(samples) >= max_samples:
                    break
            if max_samples and len(samples) >= max_samples:
                break

        self._samples = samples
        logger.info(f"[{split}] Built {len(samples)} training pairs")

    # ── PyG Dataset interface ─────────────────────────────────────────────────

    def len(self):
        return len(self._samples)

    def get(self, idx: int) -> "Data":
        row_t, row_t1 = self._samples[idx]

        lat = float(row_t["lat"])
        lon = float(row_t["lon"])
        i, j = _latlon_to_ij(lat, lon, self.n_lat, self.n_lon)

        # 3×3 patch centred on buoy (9 nodes max)
        patch_nodes, patch_feats = [], []
        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for di, dj in directions:
            ni = i + di
            nj = (j + dj) % self.n_lon
            if not (0 <= ni < self.n_lat):
                continue
            if not self.land_mask[ni, nj]:
                continue
            lat_n = float(self.lats_grid[ni])
            lon_n = float(self.lons_grid[nj])
            u_c  = float(self.currents[ni, nj, 0])
            v_c  = float(self.currents[ni, nj, 1])
            u_w  = float(self.winds[ni, nj, 0])
            v_w  = float(self.winds[ni, nj, 1])
            sst  = 0.0   # SST not in current pipeline; set to 0 placeholder
            den  = float(self.density_map[ni, nj])
            stk  = float(self.stokes[ni, nj])
            patch_nodes.append((ni, nj))
            patch_feats.append([lat_n, lon_n, u_c, v_c, u_w, v_w, sst, den, stk])

        if len(patch_nodes) < 2:
            # Degenerate patch — return a single-node dummy graph
            x = torch.zeros(1, 9, dtype=torch.float)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr  = torch.zeros(0, 3, dtype=torch.float)
            y = torch.tensor([0.0], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        x = torch.tensor(patch_feats, dtype=torch.float)  # (K, 9)
        K = len(patch_nodes)

        # Build edges between all pairs in patch
        src_list, dst_list, eattr_list = [], [], []
        from src.utils.geo_utils import haversine, bearing, current_alignment
        for a in range(K):
            for b in range(K):
                if a == b:
                    continue
                lat_a, lon_a = self.lats_grid[patch_nodes[a][0]], self.lons_grid[patch_nodes[a][1]]
                lat_b, lon_b = self.lats_grid[patch_nodes[b][0]], self.lons_grid[patch_nodes[b][1]]
                d_km = haversine(lat_a, lon_a, lat_b, lon_b)
                b_deg = bearing(lat_a, lon_a, lat_b, lon_b)
                u_a = float(self.currents[patch_nodes[a][0], patch_nodes[a][1], 0])
                v_a = float(self.currents[patch_nodes[a][0], patch_nodes[a][1], 1])
                align = current_alignment(u_a, v_a, b_deg)
                src_list.append(a)
                dst_list.append(b)
                eattr_list.append([d_km / 100.0, b_deg / 360.0, (align + 1) / 2.0])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(eattr_list, dtype=torch.float)

        # Label: normalize observed displacement to [0, 1] as density proxy
        dlat = float(row_t1["lat"]) - lat
        dlon = float(row_t1["lon"]) - lon
        disp_norm = float(np.clip(np.sqrt(dlat**2 + dlon**2) / 5.0, 0, 1))
        y = torch.tensor([disp_norm], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
