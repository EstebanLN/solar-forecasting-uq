#!/usr/bin/env python
"""Standalone, memory-safe SARIMA grid search for raw hourly GHI.

Same computation as notebook 05b cell-12, extracted to a script because
running 144 sequential SARIMAX fits inside a single long-lived process
(e.g. a Jupyter kernel) accumulates RSS via allocator fragmentation and
can push the system into OOM when run alongside the GPU training pipelines.

Mitigations here:
  - gc.collect() + malloc_trim(0) after every fit to return freed memory to the OS
  - results written incrementally to CSV (crash-safe, resumable view of progress)
  - intended to be run under `ulimit -v` so a runaway process dies cleanly
    instead of starving the rest of the system
"""
from __future__ import annotations

import ctypes
import gc
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

SITE = sys.argv[1] if len(sys.argv) > 1 else "elpaso"

GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
MANIFEST_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / f"sarima_grid_search_{SITE}.csv"

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def release_memory() -> None:
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)


print(f"Site: {SITE}")
ground = pd.read_parquet(GROUND_DIR / f"ground_10min_utc_{SITE}.parquet")
ghi_hourly = ground["ghi"].resample("1h").mean().dropna()

man_dir = MANIFEST_DIR / SITE / "h1"
train_man = pd.read_parquet(man_dir / "manifest_train.parquet")
t_labels = pd.to_datetime(train_man["t_label"], utc=True)
train_start, train_end = t_labels.min(), t_labels.max()

ghi_train = ghi_hourly.loc[(ghi_hourly.index >= train_start) & (ghi_hourly.index <= train_end)]
print(f"Training period: {train_start.date()} -> {train_end.date()}")
print(f"ghi_train: {len(ghi_train):,} hourly observations")

del ground, ghi_hourly, train_man, t_labels
release_memory()

P_RANGE = [0, 1]
Q_RANGE = [0, 1]
p_RANGE = [0, 1, 2, 3]
q_RANGE = [0, 1, 2, 3]
D, d = 1, 1
m = 24

ghi_search = ghi_train.iloc[-18 * 30 * 24 :]
print(f"Grid search on {len(ghi_search):,} obs (last 18 months of training)")

combos = [
    (p, q, P, Q)
    for p, q, P, Q in itertools.product(p_RANGE, q_RANGE, P_RANGE, Q_RANGE)
    if not (p == 0 and q == 0)
]
total = len(combos)
print(f"Grid size: {total} models\n")

results: list[dict] = []
done = 0

for p, q, P, Q in combos:
    t0 = time.time()
    try:
        mod = SARIMAX(
            ghi_search,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mod.fit(disp=False, maxiter=150, low_memory=True)
        results.append(
            {
                "order": (p, d, q),
                "seasonal": (P, D, Q, m),
                "aic": fit.aic,
                "bic": fit.bic,
                "converged": fit.mle_retvals.get("converged", True),
            }
        )
        done += 1
        elapsed = time.time() - t0
        print(
            f"  [{done}/{total}] SARIMAX{(p, d, q)}x{(P, D, Q, m)}  "
            f"AIC={fit.aic:.1f}  BIC={fit.bic:.1f}  {elapsed:.0f}s",
            flush=True,
        )
        del mod, fit
    except Exception as e:
        print(f"  FAIL  SARIMAX{(p, d, q)}x{(P, D, Q, m)}: {e}", flush=True)
    finally:
        release_memory()

    # Write progress incrementally so partial results survive a crash/kill.
    if results:
        grid_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
        grid_df_serializable = grid_df.copy()
        grid_df_serializable["order"] = grid_df_serializable["order"].astype(str)
        grid_df_serializable["seasonal"] = grid_df_serializable["seasonal"].astype(str)
        grid_df_serializable.to_csv(OUT_CSV, index=False)

print(f"\nDone. {done}/{total} models fitted. Saved to: {OUT_CSV}")
print("\nTop 10 by AIC:")
print(pd.DataFrame(results).sort_values("aic").head(10).to_string(index=False))
