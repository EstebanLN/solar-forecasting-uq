"""Markov-chain diagnostics: split-R-hat and bulk-ESS.

Standard implementations (Gelman et al., BDA3; Vehtari et al. 2021 for split-R-hat;
Geyer's initial monotone sequence for ESS). No external dependency beyond numpy,
so they run identically on both machines and inside the SGLD runners.

Both functions accept an array of shape (M, N) for a single scalar across M chains
of N draws, or (M, N, D) for D monitored scalars at once; they return a float or a
(D,) array respectively.
"""
from __future__ import annotations

import numpy as np


def _as_3d(chains: np.ndarray) -> tuple[np.ndarray, bool]:
    if chains.ndim == 2:
        return chains[:, :, None], True
    if chains.ndim == 3:
        return chains, False
    raise ValueError(f"chains must be (M,N) or (M,N,D), got shape {chains.shape}")


def split_rhat(chains: np.ndarray) -> np.ndarray | float:
    """Split-R-hat per monitored scalar. Each chain is split in half (→ 2M
    half-chains) before the between/within variance ratio, which makes the
    statistic sensitive to non-stationarity WITHIN a chain (a drifting chain
    gets R-hat >> 1 even if all chains drift the same way)."""
    X, squeeze = _as_3d(chains)
    M, N, D = X.shape
    n = N // 2
    if n < 2:
        out = np.full(D, np.nan)
        return float(out[0]) if squeeze else out
    halves = np.concatenate([X[:, :n, :], X[:, n:2 * n, :]], axis=0)  # (2M, n, D)
    m = halves.shape[0]
    chain_means = halves.mean(axis=1)                 # (2M, D)
    chain_vars = halves.var(axis=1, ddof=1)           # (2M, D)
    grand_mean = chain_means.mean(axis=0)             # (D,)
    B = n / (m - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = chain_vars.mean(axis=0)
    W = np.where(W <= 0, np.nan, W)
    var_plus = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_plus / W)
    return float(rhat[0]) if squeeze else rhat


def ess_bulk(chains: np.ndarray) -> np.ndarray | float:
    """Bulk effective sample size per monitored scalar, via Geyer's initial
    positive + monotone sequence estimator on the multi-chain autocorrelation."""
    X, squeeze = _as_3d(chains)
    M, N, D = X.shape
    ess = np.empty(D)
    for d in range(D):
        x = X[:, :, d]                                # (M, N)
        chain_means = x.mean(axis=1)
        W = x.var(axis=1, ddof=1).mean()
        B = N * chain_means.var(ddof=1) if M > 1 else 0.0
        var_plus = (N - 1) / N * W + (B / N if M > 1 else 0.0)
        if not np.isfinite(var_plus) or var_plus <= 0:
            ess[d] = float(M * N)
            continue
        acov = np.zeros(N)
        for mi in range(M):
            xi = x[mi] - chain_means[mi]
            f = np.fft.rfft(xi, n=2 * N)
            ac = np.fft.irfft(f * np.conjugate(f))[:N].real
            acov += ac / N
        acov /= M
        rho = 1.0 - (W - acov) / var_plus
        rho[0] = 1.0
        t, rho_sum, prev_pair = 1, 0.0, np.inf
        while t + 1 < N:
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            pair = min(pair, prev_pair)               # monotone
            rho_sum += pair
            prev_pair = pair
            t += 2
        tau = 1.0 + 2.0 * rho_sum
        ess[d] = float(M * N / max(tau, 1e-8))
    return float(ess[0]) if squeeze else ess
