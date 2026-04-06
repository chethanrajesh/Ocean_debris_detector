"""
train_gnn.py
Training loop for OceanDebrisGNN.

Loss:   MSE(prediction, label) + λ_smooth × spatial_smoothness_loss
Optim:  Adam, lr=1e-4, cosine annealing scheduler
Epochs: 200 with early stopping on validation MSE (patience=15)
Output: models/gnn_checkpoint.pt

Usage
-----
  python -m src.models.train_gnn [--epochs 200] [--batch-size 32] [--lr 1e-4]
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent
CHECKPOINT = MODELS_DIR / "gnn_checkpoint.pt"
LAMBDA_SMOOTH = 0.1   # spatial smoothness regularisation weight


def spatial_smoothness_loss(
    pred: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise sharp density discontinuities between adjacent ocean nodes.
    L_smooth = mean( (pred[src] - pred[dst])^2 )
    """
    if edge_index.shape[1] == 0:
        return torch.tensor(0.0, device=pred.device)
    src = edge_index[0]
    dst = edge_index[1]
    return ((pred[src] - pred[dst]) ** 2).mean()


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)          # (N_batch_nodes,)
        # Align prediction shape with labels
        y = batch.y.squeeze(-1)      # (N_graphs,) — one label per graph

        # Map per-node predictions to per-graph predictions (mean pooling)
        from torch_geometric.nn import global_mean_pool
        pred_graph = global_mean_pool(pred, batch.batch)   # (N_graphs,)

        mse = nn.MSELoss()(pred_graph, y)
        smooth = spatial_smoothness_loss(pred, batch.edge_index)
        loss = mse + LAMBDA_SMOOTH * smooth
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def val_epoch(model, loader, device) -> float:
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y    = batch.y.squeeze(-1)
            from torch_geometric.nn import global_mean_pool
            pred_graph = global_mean_pool(pred, batch.batch)
            total_mse += float(nn.MSELoss()(pred_graph, y).item())
    return total_mse / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train OceanDebrisGNN")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--max-samples",type=int,   default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # ── Imports requiring torch-geometric ─────────────────────────────────────
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        raise ImportError("torch-geometric is required. Install with: pip install torch-geometric")

    from src.models.buoy_data_loader import BuoyGraphDataset
    from src.models.density_predictor import OceanDebrisGNN

    logger.info("Loading datasets...")
    train_ds = BuoyGraphDataset("train", max_samples=args.max_samples)
    val_ds   = BuoyGraphDataset("val",   max_samples=args.max_samples // 8 if args.max_samples else None)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty. Run the data pipeline first.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model     = OceanDebrisGNN().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    best_val_mse = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mse    = val_epoch(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | Val MSE: {val_mse:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), str(CHECKPOINT))
            logger.info(f"  ✓ New best Val MSE {best_val_mse:.6f} — checkpoint saved")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    logger.info(f"Training complete. Best Val MSE: {best_val_mse:.6f}")
    logger.info(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
