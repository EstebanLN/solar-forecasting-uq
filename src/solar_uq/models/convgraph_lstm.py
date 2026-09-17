"""Hybrid satellite-only spatial encoder: convolutional (exact site crop) +
GraphSAGE (K random neighbouring patches, one node each) → concat → LSTM.

Idea (research-seminar addition, 2026-09): the exact 16x16 crop over the site is
encoded convolutionally; K neighbouring 16x16 patches — sampled at random (no
fixed set) from a non-overlapping pool around the site, one GraphSAGE node each —
capture the surrounding spatial context. The two per-frame embeddings are
concatenated before the temporal LSTM.

Input : center_seq (B, L, C, P, P),  neigh_seq (B, L, K, C, P, P)
Output: scalar prediction per sample (B,)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .resnet_lstm import SmallResNetEncoder
from .graphsage_lstm import GraphSAGELayer


def _fully_connected_edge_index(k: int) -> torch.Tensor:
    """Directed edges u->v for every ordered pair u != v over k nodes: (2, k*(k-1))."""
    src, dst = [], []
    for u in range(k):
        for v in range(k):
            if u != v:
                src.append(u)
                dst.append(v)
    return torch.tensor([src, dst], dtype=torch.long)


class ConvGraphLSTM(nn.Module):
    def __init__(
        self,
        in_ch: int = 16,
        k_neighbors: int = 8,
        # center convolutional encoder
        base_c: int = 32,
        emb_c: int = 128,
        # per-neighbour convolutional encoder (node featuriser)
        base_n: int = 16,
        emb_n: int = 64,
        # graph over the K neighbour nodes
        hidden_g: int = 64,
        n_sage_layers: int = 2,
        concat_agg: bool = True,
        # temporal
        hidden_t: int = 128,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k_neighbors

        self.center_enc = SmallResNetEncoder(in_ch=in_ch, base=base_c, emb_dim=emb_c)
        self.neigh_enc  = SmallResNetEncoder(in_ch=in_ch, base=base_n, emb_dim=emb_n)
        self.node_bn    = nn.BatchNorm1d(emb_n)

        sage, dim = [], emb_n
        for _ in range(n_sage_layers):
            sage.append(GraphSAGELayer(dim, hidden_g, concat_agg=concat_agg))
            dim = hidden_g
        self.sage = nn.ModuleList(sage)
        self.register_buffer("base_edge", _fully_connected_edge_index(k_neighbors))

        fused = emb_c + hidden_g
        self.emb_norm = nn.LayerNorm(fused)
        self.lstm = nn.LSTM(
            input_size=fused, hidden_size=hidden_t, num_layers=n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_t, hidden_t), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_t, 1),
        )

    def _batched_edges(self, n_graphs: int, device) -> torch.Tensor:
        """Disjoint union of `n_graphs` copies of the K-node graph: (2, n_graphs*E)."""
        e = self.base_edge.to(device)
        E = e.size(1)
        off = (torch.arange(n_graphs, device=device) * self.k).view(-1, 1)   # (G,1)
        batched = e.unsqueeze(0) + off.unsqueeze(-1)                           # (G,2,E)
        return batched.permute(1, 0, 2).reshape(2, n_graphs * E).contiguous()

    def forward(self, center_seq: torch.Tensor, neigh_seq: torch.Tensor) -> torch.Tensor:
        B, L, C, P, P2 = center_seq.shape
        K = neigh_seq.shape[2]

        # Center convolutional branch (per frame)
        ce = self.center_enc(center_seq.reshape(B * L, C, P, P2))            # (B*L, emb_c)

        # Neighbour graph branch (per frame): each neighbour patch -> one node
        nf = self.neigh_enc(neigh_seq.reshape(B * L * K, C, P, P2))          # (B*L*K, emb_n)
        nf = self.node_bn(nf)
        G  = B * L
        ei = self._batched_edges(G, nf.device)                               # (2, G*E)
        h  = nf
        for layer in self.sage:
            h = layer(h, ei)                                                 # (G*K, hidden_g)
        ge = h.reshape(G, K, -1).mean(dim=1)                                 # mean readout (G, hidden_g)

        # Concatenate the two embeddings before the temporal encoder
        fused = torch.cat([ce, ge], dim=-1)                                  # (B*L, emb_c+hidden_g)
        fused = self.emb_norm(fused).reshape(B, L, -1)                       # (B, L, fused)
        out, _ = self.lstm(fused)
        return self.head(out[:, -1]).squeeze(-1)                             # (B,)
