"""MLP baseline: spatial-average-pool each patch frame, then fully-connected network."""
from __future__ import annotations

import torch
import torch.nn as nn


class FlatMLP(nn.Module):
    """
    Input : (B, L, C, P, P) from PatchSeqDataset
    Steps : spatial mean-pool → (B, L, C), flatten → (B, L*C), MLP → (B,)
    """

    def __init__(
        self,
        L: int,
        C: int = 16,
        n_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = L * C
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, P, P)
        x = x.mean(dim=(-2, -1))        # (B, L, C) — spatial avg pool
        x = x.reshape(x.shape[0], -1)   # (B, L*C)
        return self.net(x).squeeze(-1)  # (B,)
