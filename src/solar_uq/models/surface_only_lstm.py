"""Surface-only LSTM: the fusion architecture with the satellite branch removed.

Deliberately mirrors FusionResNetLSTM downstream of the fusion point --- same
TabularProjector, same LayerNorm, same LSTM, same head --- so that the gap
between this model and the fusion model isolates the satellite contribution and
nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fusion.tabular_projector import TabularProjector


class SurfaceOnlyLSTM(nn.Module):
    def __init__(self, d_tab: int, emb_dim: int = 192, hidden_t: int = 128,
                 n_lstm_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.tab_proj = TabularProjector(d_tab, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_t,
                            num_layers=n_lstm_layers, batch_first=True,
                            dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_t, hidden_t), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_t, 1),
        )

    def forward(self, tab_seq: torch.Tensor) -> torch.Tensor:
        z = self.emb_norm(self.tab_proj(tab_seq))     # (B, L, emb)
        out, _ = self.lstm(z)
        return self.head(out[:, -1]).squeeze(-1)       # (B,)
