"""TabularProjector: project surface tabular features into the satellite embedding space."""
from __future__ import annotations

import torch
import torch.nn as nn


class TabularProjector(nn.Module):
    """Project a tabular feature vector of dimension d_tab into d_emb.

    Architecture:
        Linear(d_tab → hidden) → LayerNorm(hidden) → ReLU → Linear(hidden → d_emb)

    The output is designed to be **residually added** to the satellite embedding
    produced by SmallResNetEncoder or GraphSAGE mean-readout:

        fused_ℓ = e_ℓ + projector(tab_ℓ)

    where e_ℓ ∈ R^{d_emb} and tab_ℓ ∈ R^{d_tab}.

    Args:
        d_tab  : number of input tabular features (e.g. 9 with DEFAULT_FEATURE_COLS)
        d_emb  : satellite embedding dimension to project into
        hidden : hidden layer width (default 64)
    """

    def __init__(self, d_tab: int, d_emb: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_tab, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_emb),
        )

    def forward(self, tab_seq: torch.Tensor) -> torch.Tensor:
        """Project tabular features into embedding space.

        Args:
            tab_seq : (B, L, d_tab)

        Returns:
            (B, L, d_emb) — projection to add to satellite embedding
        """
        return self.net(tab_seq)
