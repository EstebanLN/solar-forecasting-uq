"""Treat an SGLD run as a Markov chain and report convergence/performance
diagnostics (advisor's suggestion): trace, autocorrelation + integrated
autocorrelation time, effective sample size (ESS), and Geweke stationarity z.

Scalar chain used = per-epoch daytime val RMSE from train_log.json (a natural
scalar summary of the chain). Point it at an SGLD run dir; with the
decreasing-step-size schedule the chain should be stationary (small Geweke |z|,
ESS not tiny) — a fixed-step drifting chain fails both.

Usage:
    .venv/bin/python scripts/09b_sgld_markov_diagnostics.py <run_dir>
    .venv/bin/python scripts/09b_sgld_markov_diagnostics.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def autocorr(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    max_lag = max_lag or (n - 1)
    v = np.var(x) + 1e-12
    return np.array([1.0 if k == 0 else float((x[:n-k] * x[k:]).mean() / v)
                     for k in range(max_lag + 1)])


def ess(x: np.ndarray) -> float:
    """Effective sample size via initial-positive-sequence autocorr sum."""
    n = len(x); rho = autocorr(x, max_lag=n - 1)
    s = 0.0
    for k in range(1, len(rho)):
        if rho[k] <= 0:  # truncate at first non-positive autocorrelation
            break
        s += rho[k]
    return float(n / (1.0 + 2.0 * s))


def geweke_z(x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    n = len(x); a = x[:max(2, int(first * n))]; b = x[int((1 - last) * n):]
    va, vb = np.var(a) / len(a) + 1e-12, np.var(b) / len(b) + 1e-12
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def diagnose(chain: np.ndarray, label: str) -> dict:
    chain = np.asarray(chain, float)
    n = len(chain); e = ess(chain); z = geweke_z(chain)
    rho1 = autocorr(chain, 1)[1] if n > 1 else float("nan")
    d = {"label": label, "n": n, "ess": round(e, 2),
         "ess_per_sample": round(e / n, 3), "lag1_autocorr": round(rho1, 3),
         "geweke_z": round(z, 2), "stationary_|z|<2": bool(abs(z) < 2.0)}
    return d


def from_run(run_dir: Path) -> np.ndarray:
    tl = json.load(open(run_dir / "train_log.json"))
    rows = tl if isinstance(tl, list) else tl.get("train_log", [])
    key = "val_rmse_day" if rows and "val_rmse_day" in rows[0] else "val_rmse"
    return np.array([r[key] for r in rows if key in r], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0)
        # AR(1) phi=0.9 (sticky → low ESS) and iid (ESS≈N) sanity checks
        ar = np.zeros(500)
        for t in range(1, 500):
            ar[t] = 0.9 * ar[t-1] + rng.normal()
        iid = rng.normal(size=500)
        drift = np.linspace(0, 10, 500) + rng.normal(0, 0.3, 500)  # non-stationary
        for c, lbl in [(ar, "AR(1) phi=0.9"), (iid, "iid"), (drift, "drift")]:
            print(diagnose(c, lbl))
        return
    rd = Path(a.run_dir)
    chain = from_run(rd)
    print(f"run: {rd.name}")
    print(diagnose(chain, "val_rmse_day chain"))


if __name__ == "__main__":
    main()
