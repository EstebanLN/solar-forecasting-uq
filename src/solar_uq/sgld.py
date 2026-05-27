"""SGLD optimizer — Welling & Teh (2011), "Bayesian Learning via Stochastic Gradient Langevin Dynamics".

Update rule per parameter at step t:
    θ_{t+1} = θ_t  -  lr * (∇loss + weight_decay * θ_t)  +  N(0, 2·lr)

At sufficiently small lr, the Markov chain samples from
    p(θ | D) ∝ exp(-loss(θ)) · N(0, 1/weight_decay)
i.e., a Gaussian posterior (L2 prior) given MSE likelihood.

Design notes:
- AMP is NOT used: noise injection must happen in the same numerical regime as
  the gradient, so we operate entirely in fp32.
- No LR scheduler: unlike Adam, SGLD requires a near-constant (or very slowly
  decaying) step size to maintain ergodicity. Decaying to zero recovers SGD.
- weight_decay encodes the Gaussian prior precision (σ² = 1/weight_decay).
  It is separate from Adam's weight_decay conceptually but numerically equivalent.
- n_train scaling: we do NOT rescale the gradient by N/batch_size here;
  the LR absorbs that factor, consistent with the existing train_one_model loop.
"""
from __future__ import annotations

import torch


class SGLD(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    Args:
        params:        model parameters (same interface as any torch Optimizer)
        lr:            SGLD step size ε. Typically 1e-5 to 1e-4 — much smaller
                       than Adam's LR. Rule of thumb: ε ≈ adam_lr * 0.01.
        weight_decay:  L2 prior precision. Use the value found by Optuna.
    """

    def __init__(self, params, lr: float = 1e-5, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            noise_std = (2.0 * lr) ** 0.5  # N(0, 2ε) per Welling & Teh

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if wd != 0.0:
                    # Gaussian prior gradient: -∇log p(θ) = weight_decay * θ
                    d_p = d_p + wd * p.data
                # Gradient descent + Langevin diffusion
                p.data.add_(-lr * d_p + noise_std * torch.randn_like(p.data))

        return loss
