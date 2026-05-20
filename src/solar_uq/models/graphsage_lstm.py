"""Improved GraphSAGE spatial encoder + LSTM temporal encoder.

Key improvements over the original baseline:
  - BatchNorm1d on input node features (channel normalization before first SAGE layer).
  - Standard SAGEConv aggregation: concat([self, neighbour_mean]) + linear,
    controlled by ``concat_agg`` (default True).  This doubles effective
    parameters without changing depth and gives gradients a richer signal.
  - Configurable number of SAGE layers (``n_sage_layers``, default 2).
  - LayerNorm on the graph-level embedding before the LSTM (same as ResNet).
  - edge_index stored as a registered buffer so the model has the same
    ``forward(x_seq)`` signature as ResNetLSTM — no external edge_index
    needed at call sites.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def build_edge_index_8n(patch: int) -> torch.Tensor:
    """8-neighbourhood edge index for a (patch × patch) pixel grid.

    Returns a (2, E) LongTensor where edge (u, v) means u → v.
    Kept for backward compatibility with older checkpoints.
    """
    edges = []
    for rr in range(patch):
        for cc in range(patch):
            u = rr * patch + cc
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < patch and 0 <= c2 < patch:
                        edges.append((u, r2 * patch + c2))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)


def build_weighted_edge_index(patch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """8-neighbourhood weighted edge index for a (patch × patch) pixel grid.

    Weights are inverse Euclidean distance on the 2-D grid:
      - Cardinal neighbours (distance 1.0)  → weight 1.000
      - Diagonal neighbours (distance √2)   → weight 1/√2 ≈ 0.707

    Returns:
        edge_index  : (2, E) LongTensor
        edge_weight : (E,)  FloatTensor — 1 / d(u, v)
    """
    return build_weighted_knn_edge_index(patch, k=8)


def build_weighted_knn_edge_index(patch: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k-nearest-neighbour weighted edge index for a (patch × patch) pixel grid.

    For each pixel u, connects its k nearest pixels by Euclidean distance on
    the 2-D grid.  Weights are inverse distance: w(u,v) = 1 / d(u,v).

    Typical k values and the neighbours they include for an interior pixel:
      k= 4  → 4 cardinal neighbours      (d = 1.0)
      k= 8  → + 4 diagonal neighbours    (d = √2  ≈ 1.41)
      k=12  → + 4 two-step cardinal      (d = 2.0)
      k=16  → + 4 of the 8 knight moves  (d = √5  ≈ 2.24)

    Border/corner pixels that have fewer than k neighbours at short range
    automatically connect to farther pixels to reach exactly k neighbours.

    Returns:
        edge_index  : (2, N*k) LongTensor
        edge_weight : (N*k,)   FloatTensor — 1 / d(u, v)
    """
    N = patch * patch
    coords = [(r, c) for r in range(patch) for c in range(patch)]

    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for u, (ru, cu) in enumerate(coords):
        dists = sorted(
            ((((ru - rv) ** 2 + (cu - cv) ** 2) ** 0.5), v)
            for v, (rv, cv) in enumerate(coords)
            if v != u
        )
        for d, v in dists[:k]:
            edges.append((u, v))
            weights.append(1.0 / d)

    edge_index  = torch.tensor(edges,   dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def _batch_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Replicate edge_index for each graph in the batch (disjoint union).

    Returns (2, E * batch_size).
    """
    edge_index = edge_index.to(device)
    E = edge_index.size(1)
    offsets = (torch.arange(batch_size, device=device) * num_nodes).view(-1, 1)  # (B, 1)
    batched = edge_index.unsqueeze(0) + offsets.unsqueeze(-1)                    # (B, 2, E)
    return batched.permute(1, 0, 2).reshape(2, batch_size * E).contiguous()      # (2, B*E)


def _batch_edge_weight(
    edge_weight: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Replicate edge_weight for each graph in the batch.

    Returns (E * batch_size,).
    """
    return edge_weight.to(device).repeat(batch_size)


# ---------------------------------------------------------------------------
# GraphSAGE layer
# ---------------------------------------------------------------------------

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE convolution layer.

    When ``concat_agg=True`` (default, standard SAGEConv):
        out = Linear(concat([x_self, mean(x_neighbours)]))
    When ``concat_agg=False`` (original additive form):
        out = Linear_self(x_self) + Linear_nei(mean(x_neighbours))

    In both cases followed by ReLU.
    """

    def __init__(self, in_dim: int, out_dim: int, concat_agg: bool = True):
        super().__init__()
        self.concat_agg = concat_agg
        if concat_agg:
            self.lin = nn.Linear(in_dim * 2, out_dim)
        else:
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_nei  = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (N_total, F),  edge_index: (2, E_total),  edge_weight: (E_total,) or None
        src, dst = edge_index[0], edge_index[1]
        N, F = x.shape

        if edge_weight is not None:
            w = edge_weight.to(x.device).view(-1, 1)        # (E, 1)
            nei_wsum = torch.zeros((N, F), device=x.device, dtype=x.dtype)
            nei_wsum.index_add_(0, dst, x[src] * w)         # Σ w·x_j  per dst
            w_total = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
            w_total.index_add_(0, dst, w)                    # Σ w      per dst
            nei_mean = nei_wsum / (w_total + 1e-6)
        else:
            nei_sum = torch.zeros((N, F), device=x.device, dtype=x.dtype)
            nei_sum.index_add_(0, dst, x[src])
            nei_cnt = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
            nei_cnt.index_add_(0, dst, torch.ones((len(dst), 1), device=x.device, dtype=x.dtype))
            nei_mean = nei_sum / (nei_cnt + 1e-6)

        if self.concat_agg:
            out = self.lin(torch.cat([x, nei_mean], dim=-1))
        else:
            out = self.lin_self(x) + self.lin_nei(nei_mean)

        return torch.relu(out)


# ---------------------------------------------------------------------------
# GraphSAGE + LSTM model
# ---------------------------------------------------------------------------

class GraphSAGE_LSTM(nn.Module):
    """
    GraphSAGE spatial encoder (per time step) → mean readout → LayerNorm
    → LSTM temporal encoder → MLP head.

    Input : x_seq (B, L, N=P*P, C=16)
    Output: scalar prediction per sample (B,)

    The ``edge_index`` is stored as a registered buffer so the model exposes
    the same ``forward(x_seq)`` signature as ResNetLSTM.
    """

    def __init__(
        self,
        in_dim: int = 16,
        hidden_g: int = 64,
        n_sage_layers: int = 2,
        hidden_t: int = 64,
        n_lstm_layers: int = 1,
        dropout_head: float = 0.0,
        input_bn: bool = True,
        concat_agg: bool = True,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Optional input channel normalisation
        self.input_bn: nn.Module = (
            nn.BatchNorm1d(in_dim) if input_bn else nn.Identity()
        )

        # SAGE layers
        sage_list = []
        dim = in_dim
        for _ in range(n_sage_layers):
            sage_list.append(GraphSAGELayer(dim, hidden_g, concat_agg=concat_agg))
            dim = hidden_g
        self.sage_layers = nn.ModuleList(sage_list)

        # Graph-level embedding normalisation before temporal encoder
        self.emb_norm = nn.LayerNorm(hidden_g)

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=hidden_g,
            hidden_size=hidden_t,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout_head if n_lstm_layers > 1 else 0.0,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_t, 1),
        )

        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.edge_index: Optional[torch.Tensor] = None

        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight: Optional[torch.Tensor] = None

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq       : (B, L, N, C)
            edge_index  : (2, E) — optional override; falls back to self.edge_index
            edge_weight : (E,)  — optional override; falls back to self.edge_weight
        """
        B, L, N, C = x_seq.shape
        device = x_seq.device

        ei = edge_index  if edge_index  is not None else self.edge_index
        ew = edge_weight if edge_weight is not None else self.edge_weight

        be = _batch_edge_index(ei, batch_size=B, num_nodes=N, device=device)
        bw = _batch_edge_weight(ew, batch_size=B, device=device) if ew is not None else None

        embeds = []
        for t in range(L):
            x = x_seq[:, t].reshape(B * N, C)    # (B*N, C)

            # Input normalisation
            x = self.input_bn(x)

            # SAGE stack — pass edge weights for weighted aggregation
            for layer in self.sage_layers:
                x = layer(x, be, bw)

            h = x.view(B, N, -1)                  # (B, N, hidden_g)
            g = h.mean(dim=1)                      # (B, hidden_g)  — mean readout
            embeds.append(g)

        z = torch.stack(embeds, dim=1)             # (B, L, hidden_g)
        z = self.emb_norm(z)

        out, _ = self.lstm(z)                      # (B, L, hidden_t)
        last   = out[:, -1]                        # (B, hidden_t)
        return self.head(last).squeeze(-1)          # (B,)
