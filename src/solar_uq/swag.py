"""SWAG: a Gaussian over weights fitted from the SGD trajectory.

Maddox et al., "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
(NeurIPS 2019, arXiv:1902.02476). While SGD runs at a constant learning rate
from a trained solution, keep the running first and second moments of the
weights and the last K deviations from the running mean. The posterior
approximation is N(mean, 1/2 * (diag + D D^T / (K-1))).

MultiSWAG (Wilson & Izmailov 2020) is the equal-weight mixture of SWAG
Gaussians fitted from independent training runs; here the independent runs are
the per-seed tuned models that already exist, so it costs one SWAG per seed.

Why try it here: SGLD does not mix on these networks (split-R-hat ~1.2 with a
bulk-ESS of ~8), and a deep ensemble gives only as many posterior modes as
there are seeds. SWAG gives a local Gaussian around each mode for the price of
a few extra epochs.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SWAG:
    def __init__(self, model: nn.Module, rank: int = 20):
        self.params: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self.n = 0
        self.rank = rank
        flat = self._flat()
        self.mean = torch.zeros_like(flat)
        self.sq_mean = torch.zeros_like(flat)
        self.dev: List[torch.Tensor] = []

    def _flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params]).double().cpu()

    def collect(self) -> None:
        """Call once per epoch of the constant-learning-rate phase."""
        w = self._flat()
        self.n += 1
        self.mean += (w - self.mean) / self.n
        self.sq_mean += (w * w - self.sq_mean) / self.n
        self.dev.append((w - self.mean).float())
        if len(self.dev) > self.rank:
            self.dev.pop(0)

    def _set(self, flat: torch.Tensor) -> None:
        i = 0
        for p in self.params:
            k = p.numel()
            p.data.copy_(flat[i:i + k].reshape(p.shape).to(p.device, p.dtype))
            i += k

    def set_mean(self) -> None:
        """Load the SWA solution (the mean of the Gaussian)."""
        self._set(self.mean)

    def sample(self, scale: float = 0.5, generator: torch.Generator | None = None) -> None:
        """Draw weights from the SWAG Gaussian and load them into the model.

        scale multiplies the covariance; 0.5 is the value used in the paper.
        """
        var = torch.clamp(self.sq_mean - self.mean ** 2, min=1e-30)
        z1 = torch.randn(var.shape, generator=generator, dtype=torch.float64)
        w = self.mean + (scale / 2.0) ** 0.5 * var.sqrt() * z1
        K = len(self.dev)
        if K > 1:
            D = torch.stack(self.dev, dim=1).double()                 # (P, K)
            z2 = torch.randn(K, generator=generator, dtype=torch.float64)
            w = w + (scale / (2.0 * (K - 1))) ** 0.5 * (D @ z2)
        self._set(w)


@torch.no_grad()
def bn_update(model: nn.Module, loader, device: str, two_input: bool, max_batches: int = 200) -> None:
    """Recompute BatchNorm running statistics for the weights now in the model.

    Sampled (or averaged) weights were never the weights the stored BatchNorm
    statistics were estimated with, so the statistics have to be re-estimated
    by a pass over training data. Models without BatchNorm are left untouched.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                     # cumulative average over the pass
    was = model.training
    model.train()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if two_input:
            model(batch[0].to(device), batch[1].to(device))
        else:
            model(batch[0].to(device))
    model.train(was)
