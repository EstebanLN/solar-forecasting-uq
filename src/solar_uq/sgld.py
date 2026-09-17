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
  It is a SEPARATE hyperparameter from the Optuna-tuned Adam weight_decay:
  reusing Adam's value (~1e-6..1e-3) leaves the chain effectively unconfined
  over 1000+ epochs and it diverges. Use scripts/08_sgld.py's
  --sgld_prior_precision (default 100.0) instead — see that script's help
  text for the relaxation-timescale argument and the empirical verification.
- n_train scaling: we do NOT rescale the gradient by N/batch_size here;
  the LR absorbs that factor, consistent with the existing train_one_model loop.
"""
from __future__ import annotations

import torch


class SGLD(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    Args:
        params:        model parameters (same interface as any torch Optimizer)
        lr:            SGLD step size ε. Calibrated on full-resolution data to
                       1e-7 (see scripts/08_sgld.py --sgld_lr): each epoch is
                       ~3.4k steps, so the accumulated per-epoch Langevin noise
                       is far larger than a small-/subsampled-setup heuristic
                       (ε ≈ adam_lr*0.01 ≈ 1e-5) would suggest — 1e-5 diverges,
                       1e-7 samples a stable warm-started local posterior.
        weight_decay:  L2 prior precision (NOT the Optuna-tuned Adam
                       weight_decay — see module docstring; default in
                       scripts/08_sgld.py is 100.0).
    """

    def __init__(self, params, lr: float = 1e-5, weight_decay: float = 0.0,
                 lr_final: float | None = None, total_steps: int | None = None):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        # Decreasing step size (Welling & Teh 2011): a chain with a polynomially
        # decaying epsilon_t (Sum eps = inf, Sum eps^2 < inf) converges to the
        # posterior WITHOUT a Metropolis correction, unlike a fixed step which
        # random-walks off the mode. If lr_final/total_steps are given we decay
        # geometrically lr -> lr_final over total_steps; else behave as constant.
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        lr_final=lr_final, total_steps=total_steps)
        super().__init__(params, defaults)
        self._t = 0

    def _lr_at(self, group) -> float:
        lr, lr_final, total = group["lr"], group["lr_final"], group["total_steps"]
        if lr_final is None or total is None or total <= 0:
            return lr
        frac = min(self._t / float(total), 1.0)
        return float(lr * (lr_final / lr) ** frac)   # geometric decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = self._lr_at(group)
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

        self._t += 1
        return loss
