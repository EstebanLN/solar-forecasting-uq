"""FusionGraphSAGE_LSTM: GraphSAGE spatial encoder fused with surface tabular features.

After the per-frame mean readout, the tabular projection is residually added to
the graph-level embedding before the temporal encoder:

    g_ℓ = mean_readout(GraphSAGE(nodes_ℓ))   # (B, hidden_g)
    p_ℓ = TabularProjector(tab_ℓ)              # (B, hidden_g)
    z_ℓ = LayerNorm(g_ℓ + p_ℓ)               # (B, hidden_g)
    ...stack L steps → LSTM → head → (B,)

GraphSAGE layers, utilities, and edge-index helpers are reused directly from
solar_uq.models.graphsage_lstm — no code duplication.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from solar_uq.models.graphsage_lstm import (
    GraphSAGELayer,
    _batch_edge_index,
    _batch_edge_weight,
)
from solar_uq.models.fusion.tabular_projector import TabularProjector


class FusionGraphSAGE_LSTM(nn.Module):
    """GraphSAGE+LSTM with tabular surface features fused after mean readout.

    Input:
        x_seq   : (B, L, N=P*P, C=16) — node feature sequences
        tab_seq : (B, L, D_tab)        — surface tabular features

    Output:
        (B,) — normalised GHI scalar prediction per sample

    Args:
        in_dim         : input node feature dimension (16 ABI channels)
        hidden_g       : GraphSAGE hidden / output dimension (= graph embedding dim)
        n_sage_layers  : number of GraphSAGE convolution layers
        hidden_t       : LSTM hidden size
        n_lstm_layers  : number of LSTM layers
        dropout_head   : dropout in LSTM (multi-layer) and prediction head
        input_bn       : if True, apply BatchNorm1d to input node features
        concat_agg     : if True, use SAGEConv concat aggregation (standard)
        d_tab          : tabular feature dimension
        tab_hidden     : hidden width inside TabularProjector
        edge_index     : (2, E) LongTensor registered as buffer (8-NN by default)
        edge_weight    : (E,) FloatTensor registered as buffer (optional)
        freeze_encoder : if True, freeze GraphSAGE and input_bn weights
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
        d_tab: int = 9,
        tab_hidden: int = 64,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.input_bn: nn.Module = (
            nn.BatchNorm1d(in_dim) if input_bn else nn.Identity()
        )

        sage_list = []
        dim = in_dim
        for _ in range(n_sage_layers):
            sage_list.append(GraphSAGELayer(dim, hidden_g, concat_agg=concat_agg))
            dim = hidden_g
        self.sage_layers = nn.ModuleList(sage_list)

        self.tab_proj = TabularProjector(d_tab=d_tab, d_emb=hidden_g, hidden=tab_hidden)
        self.emb_norm = nn.LayerNorm(hidden_g)

        self.lstm = nn.LSTM(
            input_size=hidden_g,
            hidden_size=hidden_t,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout_head if n_lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU(inplace=True),
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

        if freeze_encoder:
            for m in [self.input_bn, self.sage_layers]:
                for param in (m.parameters() if hasattr(m, "parameters") else []):
                    param.requires_grad_(False)

    def forward(
        self,
        x_seq: torch.Tensor,
        tab_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq       : (B, L, N, C)
            tab_seq     : (B, L, D_tab)
            edge_index  : (2, E) — optional override
            edge_weight : (E,)  — optional override

        Returns:
            (B,)
        """
        B, L, N, C = x_seq.shape
        device = x_seq.device

        ei = edge_index  if edge_index  is not None else self.edge_index
        ew = edge_weight if edge_weight is not None else self.edge_weight

        be = _batch_edge_index(ei, batch_size=B, num_nodes=N, device=device)
        bw = _batch_edge_weight(ew, batch_size=B, device=device) if ew is not None else None

        # Tabular projection — one pass over all timesteps
        p_all = self.tab_proj(tab_seq)   # (B, L, hidden_g)

        embeds = []
        for t in range(L):
            x = x_seq[:, t].reshape(B * N, C)   # (B*N, C)
            x = self.input_bn(x)
            for layer in self.sage_layers:
                x = layer(x, be, bw)
            h = x.view(B, N, -1)                 # (B, N, hidden_g)
            g = h.mean(dim=1)                    # (B, hidden_g)
            embeds.append(g)

        z_sat = torch.stack(embeds, dim=1)       # (B, L, hidden_g)
        z     = self.emb_norm(z_sat + p_all)     # (B, L, hidden_g)

        out, _ = self.lstm(z)                    # (B, L, hidden_t)
        last   = out[:, -1]                      # (B, hidden_t)
        return self.head(last).squeeze(-1)       # (B,)
