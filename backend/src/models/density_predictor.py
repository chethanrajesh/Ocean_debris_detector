"""
density_predictor.py
OceanDebrisGNN: Graph Attention Network for predicting plastic debris density.

Architecture
------------
  Input Graph State at time t
           ↓
  [Node Encoder]           Linear(9 → 128) → LayerNorm → ReLU
  [Edge Encoder]           Linear(3 → 64)  → LayerNorm → ReLU
           ↓
  [Message Passing × 6]    GATv2Conv, 8 attention heads, hidden dim 128
           ↓
  [Residual Correction]    Linear(128 → 64) → ReLU → Linear(64 → 1) → Sigmoid
           ↓
  Output: GNN correction scalar per node at time t+1

Node feature vector (dim 9):
  [lat, lon, u_current, v_current, u_wind, v_wind, sst, plastic_density_t, stokes_drift_magnitude]

Edge feature vector (dim 3):
  [distance_km, bearing_degrees, current_alignment_score]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data, Batch
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    # Provide a stub so the module can still be imported
    class GATv2Conv(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.linear = nn.Linear(a[0], a[1] * kw.get("heads", 1))
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)

NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 3
HIDDEN_DIM    = 128
EDGE_HIDDEN   = 64
N_HEADS       = 8
N_LAYERS      = 6


class NodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NODE_FEAT_DIM, HIDDEN_DIM)
        self.norm   = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(x)))


class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(EDGE_FEAT_DIM, EDGE_HIDDEN)
        self.norm   = nn.LayerNorm(EDGE_HIDDEN)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(e)))


class GATLayer(nn.Module):
    """One GATv2Conv layer with residual connection."""
    def __init__(self, in_channels: int, heads: int, edge_dim: int):
        super().__init__()
        assert in_channels % heads == 0, "in_channels must be divisible by heads"
        out_per_head = in_channels // heads
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_per_head,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_attr)
        return F.relu(self.norm(out + x))   # residual


class CorrectionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)   # (N,) in [0, 1]


class OceanDebrisGNN(nn.Module):
    """
    Hybrid GNN for ocean debris density prediction / correction.

    Forward pass:
      data.x          : (N, 9) node features
      data.edge_index : (2, E) COO edge list
      data.edge_attr  : (E, 3) edge features

    Returns:
      correction : (N,) tensor in [0, 1] — GNN density correction per node
    """

    def __init__(self):
        super().__init__()
        self.node_encoder = NodeEncoder()
        self.edge_encoder = EdgeEncoder()

        self.gat_layers = nn.ModuleList([
            GATLayer(HIDDEN_DIM, N_HEADS, EDGE_HIDDEN)
            for _ in range(N_LAYERS)
        ])

        self.head = CorrectionHead(HIDDEN_DIM)

    def forward(self, data) -> torch.Tensor:
        x         = data.x          # (N, 9)
        edge_index = data.edge_index  # (2, E)
        edge_attr  = data.edge_attr   # (E, 3)

        node_feats = self.node_encoder(x)
        edge_feats = self.edge_encoder(edge_attr)

        for layer in self.gat_layers:
            node_feats = layer(node_feats, edge_index, edge_feats)

        return self.head(node_feats)   # (N,)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "OceanDebrisGNN":
        model = cls()
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


def build_graph_data(
    node_feats: "torch.Tensor",
    edge_index:  "torch.Tensor",
    edge_attr:   "torch.Tensor",
) -> "Data":
    """Convenience helper to build a PyG Data object."""
    if not TG_AVAILABLE:
        raise ImportError("torch-geometric is required. Install it via pip.")
    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
