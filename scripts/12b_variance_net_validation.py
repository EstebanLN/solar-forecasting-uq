#!/usr/bin/env python
"""
12b_variance_net_validation.py — validate the aleatoric variance network against
a KNOWN heteroscedastic sigma(x).

Why: on the real data the variance network returned a near-constant sigma_a
(mean 302 W/m^2, std 12, corr 0.23 vs |residual|) — i.e. it was homoscedastic in
disguise. This script isolates the cause on synthetic data where the true
sigma(x) is known, and shows that the fix (exponential parameterisation + a
residual fed in normalised units) recovers it.

Setup: y = f(x) + sigma(x) * eps, eps ~ N(0,1), with f and sigma known, so the
frozen "mean network" is exactly f and the squared residual has
E[r | x] = sigma(x)^2. We then fit the Gamma variance head from
src/solar_uq/variance_net.py and compare sigma_hat(x) against sigma(x).

Two parameterisations are compared on identical data and identical training:
  * 'softplus-physical' — the previous code: softplus(raw), residual left in
    physical units (scaled by SCALE, mimicking W/m^2).
  * 'exp-normalised'    — the current code: exp(raw), residual normalised.

Usage:  .venv/bin/python scripts/12b_variance_net_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from solar_uq.variance_net import gamma_nll_loss, gamma_params, aleatoric_variance

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--epochs", type=int, default=400)
_ap.add_argument("--n", type=int, default=6000)
_ap.add_argument("--scale", type=float, default=300.0)
_A = _ap.parse_args()
SCALE = _A.scale       # stands in for the W/m^2 magnitude of the real residuals
N, EPOCHS = _A.n, _A.epochs


def softplus_params(raw, eps=1e-6):
    """The PREVIOUS parameterisation, kept here only for the comparison."""
    return (nn.functional.softplus(raw[..., 0]) + eps,
            nn.functional.softplus(raw[..., 1]) + eps)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(N, 1))
    f = 2.0 * np.sin(x[:, 0])                       # true mean
    sigma = 0.3 + 0.9 * np.abs(x[:, 0]) / 3.0       # TRUE heteroscedastic sd (0.3 -> 1.2)
    y = f + sigma * rng.standard_normal(N)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(f, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            sigma)


def fit(x, r, use_exp: bool, seed=0):
    """Train the 2-output Gamma head on squared residuals r; return sigma_hat."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    params = gamma_params if use_exp else softplus_params
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        alpha, lam = params(net(x))
        loss = gamma_nll_loss(alpha, lam, r)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        alpha, lam = params(net(x))
        return torch.sqrt(aleatoric_variance(alpha, lam)).numpy(), float(loss)


def report(name, sigma_hat, sigma_true):
    corr = float(np.corrcoef(sigma_hat, sigma_true)[0, 1])
    ratio = float(np.mean(sigma_hat / sigma_true))
    cv = float(sigma_hat.std() / sigma_hat.mean())          # 0 => constant
    cv_true = float(sigma_true.std() / sigma_true.mean())
    ok = corr > 0.8 and 0.8 < ratio < 1.25
    print(f"  {name:22s} corr={corr:+.3f}  ratio={ratio:5.2f}  "
          f"cv={cv:.3f} (true {cv_true:.3f})  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    x, f, y, sigma_true = make_data()

    print("=" * 74)
    print("VARIANCE-NET VALIDATION — known heteroscedastic sigma(x)")
    print(f"  epochs={EPOCHS} | n={N} | true sigma in [{sigma_true.min():.2f}, {sigma_true.max():.2f}] "
          f"| physical scale = {SCALE:g}")
    print("-" * 74)

    # residual from the exact frozen mean -> E[r|x] = sigma(x)^2
    r_norm = (f - y) ** 2                                    # normalised units, O(1)
    r_phys = r_norm * (SCALE ** 2)                           # physical-like units, O(1e4)

    sh_old, _ = fit(x, r_phys, use_exp=False)
    sh_old = sh_old / SCALE                                  # back to comparable units
    sh_new, _ = fit(x, r_norm, use_exp=True)

    ok_old = report("softplus-physical", sh_old, sigma_true)
    ok_new = report("exp-normalised", sh_new, sigma_true)

    print("-" * 74)
    if ok_new and not ok_old:
        print("  VERDICT: the fix works — the old parameterisation degenerates to a")
        print("           near-constant sigma, the new one recovers sigma(x).")
    elif ok_new:
        print("  VERDICT: new parameterisation PASSES.")
    else:
        print("  VERDICT: new parameterisation FAILS — do not use on real data yet.")
    print("=" * 74)
    sys.exit(0 if ok_new else 1)


if __name__ == "__main__":
    main()
