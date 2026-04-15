"""ResNet spatial encoder + LSTM temporal encoder for satellite patch sequences."""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return self.relu(out + identity)


class SmallResNetEncoder(nn.Module):
    """
    Lightweight ResNet encoder for a P×P satellite patch.

    Input : (B, in_ch, P, P)
    Output: (B, emb_dim)

    With P=16: 16 → 8 → 4 via stride-2 stages, then global average pool.
    """
    def __init__(self, in_ch: int = 16, base: int = 32, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(base,     base,     stride=1),
            BasicBlock(base,     base,     stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(base,     base * 2, stride=2),
            BasicBlock(base * 2, base * 2, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base * 2, base * 4, stride=2),
            BasicBlock(base * 4, base * 4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(base * 4, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).squeeze(-1).squeeze(-1)   # (B, base*4)
        return self.proj(h)                         # (B, emb_dim)


class ResNetLSTM(nn.Module):
    """
    SmallResNetEncoder applied per time step → LayerNorm → LSTM → MLP head.

    Input : x_seq (B, L, C=16, P, P)
    Output: scalar prediction per sample (B,)
    """
    def __init__(
        self,
        in_ch: int = 16,
        base: int = 32,
        emb_dim: int = 128,
        hidden_t: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder  = SmallResNetEncoder(in_ch=in_ch, base=base, emb_dim=emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.lstm     = nn.LSTM(input_size=emb_dim, hidden_size=hidden_t, batch_first=True)
        self.head     = nn.Sequential(
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_t, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, L, C, P, P2 = x_seq.shape

        x = x_seq.reshape(B * L, C, P, P)         # (B*L, C, P, P)
        z = self.encoder(x)                        # (B*L, emb_dim)
        z = self.emb_norm(z)
        z = z.reshape(B, L, -1)                    # (B, L, emb_dim)

        out, _ = self.lstm(z)                      # (B, L, hidden_t)
        last   = out[:, -1]                        # (B, hidden_t)
        return self.head(last).squeeze(-1)          # (B,)
