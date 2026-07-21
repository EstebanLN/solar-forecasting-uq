"""Correctness tests for the aleatoric variance network (Step 2, cooperative BNN-VE).

Run from project root:
    .venv/bin/python -m pytest tests/test_variance_net.py -v

All tests use synthetic random tensors — no real data or GPU required.
The most important one is test_gamma_nll_recovers_known_parameters: it does
NOT trust the paper's image-rendered loss equation, it verifies from first
principles that minimising our Gamma NLL against samples drawn from a KNOWN
Gamma(alpha_true, lambda_true) recovers alpha_true and lambda_true.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solar_uq.models.graphsage_lstm import GraphSAGE_LSTM, build_weighted_knn_edge_index
from solar_uq.models.mlp import FlatMLP
from solar_uq.models.resnet_lstm import ResNetLSTM
from solar_uq.variance_net import (
    aleatoric_variance,
    gamma_nll_loss,
    gamma_params,
    make_variance_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Head-swap genericity: same wrapper must work, unmodified, on all 3 archs
# ─────────────────────────────────────────────────────────────────────────────

def test_make_variance_model_resnet():
    B, L, C, P = 2, 4, 16, 8
    model = make_variance_model(ResNetLSTM(in_ch=C, base=8, emb_dim=16, hidden_t=16, n_lstm_layers=1))
    x_seq = torch.randn(B, L, C, P, P)
    out = model(x_seq)
    assert out.shape == (B, 2), f"Expected ({B},2), got {out.shape}"
    assert not torch.isnan(out).any()


def test_make_variance_model_graphsage():
    B, L, P, C = 2, 4, 8, 16
    N = P * P
    edge_index, edge_weight = build_weighted_knn_edge_index(P, k=8)
    model = make_variance_model(GraphSAGE_LSTM(
        in_dim=C, hidden_g=16, n_sage_layers=2, hidden_t=16, n_lstm_layers=1,
        edge_index=edge_index, edge_weight=edge_weight,
    ))
    x_seq = torch.randn(B, L, N, C)
    out = model(x_seq)
    assert out.shape == (B, 2), f"Expected ({B},2), got {out.shape}"
    assert not torch.isnan(out).any()


def test_make_variance_model_mlp():
    B, L, C = 2, 4, 16
    model = make_variance_model(FlatMLP(L=L, C=C, n_layers=1, hidden_dim=16))
    x_seq = torch.randn(B, L, C, 8, 8)
    out = model(x_seq)
    assert out.shape == (B, 2), f"Expected ({B},2), got {out.shape}"
    assert not torch.isnan(out).any()


def test_make_variance_model_rejects_non_scalar_head():
    """Should fail loudly (not silently produce garbage) if the assumed
    Linear(*, 1)-ending convention doesn't hold."""
    model = ResNetLSTM(in_ch=16, base=8, emb_dim=16, hidden_t=16)
    model.head[-1] = nn.Linear(16, 3)  # break the (*, 1) assumption
    try:
        make_variance_model(model)
        assert False, "Expected ValueError for a non-(*,1)-ending head"
    except ValueError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# gamma_params: softplus must keep alpha, lambda strictly positive
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma_params_always_positive():
    raw = torch.tensor([[-50.0, -50.0], [50.0, 50.0], [0.0, 0.0]])
    alpha, lam = gamma_params(raw)
    assert (alpha > 0).all() and (lam > 0).all()
    assert not torch.isnan(alpha).any() and not torch.isnan(lam).any()
    assert not torch.isinf(alpha).any() and not torch.isinf(lam).any()


# ─────────────────────────────────────────────────────────────────────────────
# The critical check: does minimising gamma_nll_loss actually recover the
# true Gamma(alpha, lambda) parameters from samples? This is what stands in
# for trusting the paper's Eq. 6 verbatim (see variance_net.py docstring).
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma_nll_recovers_known_parameters():
    torch.manual_seed(0)
    alpha_true, lambda_true = 3.0, 0.5   # mean = alpha/lambda = 6.0
    n = 20_000
    # torch.distributions.Gamma uses the same (shape, rate) convention.
    r = torch.distributions.Gamma(alpha_true, lambda_true).sample((n,))

    # Two free scalar parameters (pre-softplus), fit by gradient descent —
    # the network-free equivalent of the per-sample (alpha(x), lambda(x))
    # heads used in production.
    raw = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)

    for _ in range(2000):
        opt.zero_grad()
        alpha, lam = gamma_params(raw.unsqueeze(0).expand(n, 2))
        loss = gamma_nll_loss(alpha, lam, r)
        loss.backward()
        opt.step()

    alpha_hat, lambda_hat = gamma_params(raw.unsqueeze(0))
    alpha_hat, lambda_hat = alpha_hat.item(), lambda_hat.item()
    mean_hat = alpha_hat / lambda_hat

    assert abs(alpha_hat - alpha_true) / alpha_true < 0.1, (
        f"alpha recovered={alpha_hat:.3f}, true={alpha_true} — Gamma NLL did "
        f"not recover the true shape parameter"
    )
    assert abs(lambda_hat - lambda_true) / lambda_true < 0.1, (
        f"lambda recovered={lambda_hat:.3f}, true={lambda_true} — Gamma NLL "
        f"did not recover the true rate parameter"
    )
    assert abs(mean_hat - 6.0) / 6.0 < 0.05, (
        f"recovered mean alpha/lambda={mean_hat:.3f}, true=6.0"
    )


def test_aleatoric_variance_matches_gamma_mean():
    alpha = torch.tensor([4.0, 9.0])
    lam   = torch.tensor([2.0, 3.0])
    sigma2 = aleatoric_variance(alpha, lam)
    expected = alpha / lam
    assert torch.allclose(sigma2, expected)
