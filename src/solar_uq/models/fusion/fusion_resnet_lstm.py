"""FusionResNetLSTM: ResNet spatial encoder fused with surface tabular features.

The satellite embedding and tabular projection are combined at each time step
via residual addition before the temporal encoder:

    e_ℓ = SmallResNetEncoder(patch_ℓ)        # (B, emb_dim)
    p_ℓ = TabularProjector(tab_ℓ)             # (B, emb_dim)
    z_ℓ = LayerNorm(e_ℓ + p_ℓ)              # (B, emb_dim)
    ...stack L steps → LSTM → head → (B,)

The ResNet encoder is reused directly from solar_uq.models.resnet_lstm —
weights are shared with the satellite-only model when fine-tuning from a
checkpoint, and the architecture is identical to avoid divergence in ablations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from solar_uq.models.resnet_lstm import SmallResNetEncoder
from solar_uq.models.fusion.tabular_projector import TabularProjector


class FusionResNetLSTM(nn.Module):
    """ResNet+LSTM with tabular surface features fused by residual addition.

    Input:
        x_seq   : (B, L, C=16, P, P) — satellite patch sequence
        tab_seq : (B, L, D_tab)       — surface tabular features

    Output:
        (B,) — normalised GHI scalar prediction per sample

    Args:
        in_ch          : satellite channels (16 for ABI-MCMIP)
        base           : base channel width for SmallResNetEncoder
        emb_dim        : satellite embedding dimension
        d_tab          : tabular feature dimension (from n_tab_features())
        tab_hidden     : hidden width inside TabularProjector
        hidden_t       : LSTM hidden size
        n_lstm_layers  : number of LSTM layers
        dropout        : dropout probability in LSTM and head
        freeze_encoder : if True, freeze SmallResNetEncoder weights (useful when
                         fine-tuning from a satellite-only checkpoint)
    """

    def __init__(
        self,
        in_ch: int = 16,
        base: int = 32,
        emb_dim: int = 128,
        d_tab: int = 9,
        tab_hidden: int = 64,
        hidden_t: int = 128,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder   = SmallResNetEncoder(in_ch=in_ch, base=base, emb_dim=emb_dim)
        self.tab_proj  = TabularProjector(d_tab=d_tab, d_emb=emb_dim, hidden=tab_hidden)
        self.emb_norm  = nn.LayerNorm(emb_dim)
        self.lstm      = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_t,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_t, 1),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def forward(self, x_seq: torch.Tensor, tab_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq   : (B, L, C, P, P)
            tab_seq : (B, L, D_tab)

        Returns:
            (B,)
        """
        B, L, C, P, P2 = x_seq.shape

        # Encode all frames in one batched pass
        x_flat  = x_seq.reshape(B * L, C, P, P)          # (B*L, C, P, P)
        e_flat  = self.encoder(x_flat)                    # (B*L, emb_dim)
        e       = e_flat.reshape(B, L, -1)                # (B, L, emb_dim)

        # Project tabular features and fuse
        p       = self.tab_proj(tab_seq)                  # (B, L, emb_dim)
        z       = self.emb_norm(e + p)                    # (B, L, emb_dim)

        out, _  = self.lstm(z)                            # (B, L, hidden_t)
        last    = out[:, -1]                              # (B, hidden_t)
        return self.head(last).squeeze(-1)                # (B,)
