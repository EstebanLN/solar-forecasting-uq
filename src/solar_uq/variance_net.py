"""Aleatoric variance network — Step 2 of the cooperative BNN-VE strategy.

Following Yi & Bessa (2025, arXiv:2505.02743) "Cooperative Bayesian and
variance networks disentangle aleatoric and epistemic uncertainties":
training mean and variance jointly (standard MVE networks) causes
imbalanced gradients and overconfident variance estimates. Instead:

    Step 1 (mean network)     — already trained: the Optuna-winning backbone.
    Step 2 (variance network) — THIS MODULE. With the mean frozen, train a
                                 separate network on the squared residual
                                 r = (mu(x) - y)^2 via a Gamma likelihood,
                                 giving aleatoric variance sigma_a^2(x) = alpha(x)/lambda(x).
    Step 3 (epistemic/BNN)    — future work: warm-start SGLD from the mean
                                 network and use this fixed sigma_a^2(x) in
                                 its likelihood (not yet implemented).

Loss note
---------
The paper's Eq. 6 was read from a rendered PDF image; a cross-check against
its own reported partial derivatives (Eq. 18-19) suggested the OCR is not
fully reliable character-for-character. Rather than risk transcribing a
subtly wrong loss into training code, this module implements the *standard*
Gamma negative log-likelihood for r ~ Gamma(shape=alpha, rate=lambda):

    NLL = -alpha*log(lambda) + lgamma(alpha) - (alpha - 1)*log(r) + lambda*r

This is derivable from first principles and is consistent with the paper's
explicitly text-stated (not image-only) conclusion that the Gamma mean
alpha/lambda is the desired heteroscedastic variance (their Eq. 7).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Turn any of the project's point-forecast models into a 2-output Gamma head
# ---------------------------------------------------------------------------

def make_variance_model(model: nn.Module) -> nn.Module:
    """Mutate a freshly-built backbone in place so its final Linear layer
    outputs 2 raw values (Gamma shape/rate pre-activations) instead of 1.

    Works generically across ResNetLSTM/GraphSAGE_LSTM (attribute ``head``)
    and FlatMLP (attribute ``net``) without touching their class
    definitions: every one of these models ends its forward pass with
    ``<container>(last_hidden).squeeze(-1)``. Squeeze only removes a
    dimension of size 1, so once the final Linear has out_features=2 the
    same unmodified forward() naturally returns shape (B, 2) instead of
    (B,). No model class needs to change.
    """
    container_name = "head" if hasattr(model, "head") else "net"
    container = getattr(model, container_name)
    last_linear = container[-1]
    if not isinstance(last_linear, nn.Linear) or last_linear.out_features != 1:
        raise ValueError(
            f"Expected {container_name}[-1] to be a Linear(*, 1) layer, "
            f"got {last_linear!r}"
        )
    container[-1] = nn.Linear(last_linear.in_features, 2)
    return model


def gamma_params(raw: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """raw: (B, 2) pre-activations -> (alpha, lambda), both strictly positive."""
    alpha = nn.functional.softplus(raw[..., 0]) + eps
    lam   = nn.functional.softplus(raw[..., 1]) + eps
    return alpha, lam


def gamma_nll_loss(
    alpha: torch.Tensor,
    lam: torch.Tensor,
    r: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standard Gamma NLL for r ~ Gamma(shape=alpha, rate=lambda), mean-reduced.

    r is the squared residual (mu(x) - y)^2 from the frozen mean network,
    in the same physical units used throughout this project (W/m^2)^2.
    """
    r = r.clamp_min(eps)
    # torch.lgamma's CUDA JIT kernel currently fails on this environment
    # (torch built for cu130 but the system ships libnvrtc-builtins.so.13.1
    # only — a version-string mismatch, not a code bug); computing lgamma on
    # CPU sidesteps it at negligible cost (alpha is a (B,) tensor).
    lgamma_alpha = torch.lgamma(alpha.cpu()).to(alpha.device)
    nll = -alpha * torch.log(lam) + lgamma_alpha - (alpha - 1.0) * torch.log(r) + lam * r
    return nll.mean()


def aleatoric_variance(alpha: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """sigma_a^2(x) = alpha(x) / lambda(x) — the Gamma distribution's mean (Eq. 7)."""
    return alpha / lam
