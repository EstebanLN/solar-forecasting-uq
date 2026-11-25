"""Low-rank adapters (LoRA, Hu et al. 2022) for SGLD on a frozen network.

Advisor's proposal (2026-11-25): sample with SGLD only a set of LoRA adapters on the
last layers of the tuned model and keep everything else frozen. The hope is that the
chains then stay in one mode of the posterior, which the dense and pruned protocols
did not manage.

Each targeted ``nn.Linear`` becomes

    y = W x + b + (alpha / r) * B (A x)

with W and b frozen at the MAP, A of shape (r, in) and B of shape (out, r). B starts at
zero, so every chain starts exactly at the MAP function and the chains differ only
through the random A and the Langevin noise. The rank is capped at min(in, out): for
the scalar output layer, Linear(h, 1), a rank above 1 only adds a symmetry, since
B A then has rank 1 anyway.

"The last layers" are the last ``n_last`` ``nn.Linear`` modules in registration
order. Every architecture in the zoo ends in the same head,
Linear(h, h) -> ReLU -> Dropout -> Linear(h, 1) (FlatMLP ends in its last Linear
layers), so ``n_last=2`` means the same thing for all of them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.rank = max(1, min(rank, base.in_features, base.out_features))
        self.scale = (alpha if alpha else self.rank) / self.rank
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init as the LoRA paper's code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


def apply_lora(model: nn.Module, rank: int, n_last: int = 2, alpha: float | None = None) -> list[dict]:
    """Freeze every parameter of ``model`` and wrap its last ``n_last`` Linear layers.

    Returns one record per wrapped layer (name, shape, effective rank)."""
    for p in model.parameters():
        p.requires_grad_(False)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("model has no nn.Linear layer to adapt")
    out = []
    for name, lin in linears[-n_last:]:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(lin, rank, alpha)
        setattr(parent, child, wrapped)
        out.append({"layer": name, "in": lin.in_features, "out": lin.out_features,
                    "rank": wrapped.rank})
    return out


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".lora_A" in n or ".lora_B" in n
            or n.startswith(("lora_A", "lora_B"))]


def freeze_norm_stats(model: nn.Module) -> None:
    """Put BatchNorm layers in eval mode: with the base frozen, their running statistics
    must not drift either. Call after every ``model.train()``. LayerNorm has no running
    statistics and needs nothing; dropout stays as it is in the dense protocol."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
