"""SGLD optimizer — Welling & Teh (2011), "Bayesian Learning via Stochastic Gradient Langevin Dynamics".

Update rule per parameter at step t:
    θ_{t+1} = θ_t  -  lr * (∇loss + weight_decay * θ_t)  +  N(0, 2·lr)

At sufficiently small lr, the Markov chain samples from
    p(θ | D) ∝ exp(-loss(θ)) · N(0, 1/weight_decay)
i.e., a Gaussian posterior (L2 prior) given MSE likelihood.

Design notes:
- AMP is NOT used: noise injection must happen in the same numerical regime as
  the gradient, so we operate entirely in fp32.
- Step-size schedule: for a chain that CONVERGES to the posterior without a
  Metropolis correction, Welling & Teh (2011) require a decreasing step size
  ε_t with Σε_t = ∞ and Σε_t² < ∞. The polynomial schedule
  ε_t = ε_0 (1 + t/t₀)^(-γ) with γ ∈ (0.5, 1] satisfies both (default,
  ``schedule="poly"``); it decays slowly enough that the chain keeps exploring
  (Σε = ∞) while the accumulated noise stays finite (Σε² < ∞). A geometric
  decay (``schedule="geom"``) is kept only for backward compatibility — it has
  Σε_t < ∞, so it freezes the chain too fast to sample the posterior and is NOT
  a valid W&T schedule. A constant step (no lr_final) random-walks off the mode.
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
                 lr_final: float | None = None, total_steps: int | None = None,
                 schedule: str = "poly", gamma: float = 0.55):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if schedule not in ("poly", "geom"):
            raise ValueError(f"schedule must be 'poly' or 'geom', got {schedule!r}")
        if not (0.5 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0.5, 1] for Sum eps=inf, Sum eps^2<inf, got {gamma}")
        # Decreasing step size (Welling & Teh 2011). If lr_final/total_steps are
        # given we decay lr -> lr_final over total_steps; else behave as a
        # constant step. schedule="poly" is the valid W&T schedule (see module
        # docstring); "geom" is kept only for backward compatibility.
        defaults = dict(lr=lr, weight_decay=weight_decay, lr_final=lr_final,
                        total_steps=total_steps, schedule=schedule, gamma=gamma)
        super().__init__(params, defaults)
        self._t = 0

    def _lr_at(self, group) -> float:
        lr, lr_final, total = group["lr"], group["lr_final"], group["total_steps"]
        if lr_final is None or total is None or total <= 0:
            return lr
        if group["schedule"] == "geom":
            frac = min(self._t / float(total), 1.0)
            return float(lr * (lr_final / lr) ** frac)   # geometric (Sum eps<inf; not W&T)
        # Polynomial Robbins-Monro schedule: eps_t = lr * (1 + t/t0)^(-gamma),
        # with t0 solved so eps(total_steps) = lr_final. For gamma in (0.5, 1]
        # this satisfies Sum eps = inf and Sum eps^2 < inf (Welling & Teh 2011).
        gamma = group["gamma"]
        ratio = (lr / lr_final) ** (1.0 / gamma)         # = (1 + total/t0)
        t0 = total / (ratio - 1.0)
        t = min(self._t, total)
        return float(lr * (1.0 + t / t0) ** (-gamma))

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
