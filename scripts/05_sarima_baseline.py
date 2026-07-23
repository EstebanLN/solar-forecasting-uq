#!/usr/bin/env python
"""
05_sarima_baseline.py — SARIMA statistical benchmark for GHI forecasting.

Methodology (rolling-origin evaluation):
  1. Load the same ground-truth GHI tabular data (`ground_10min_utc_{site}.parquet`)
     used to train and evaluate every other model, and resample it to hourly
     resolution (m=24 seasonal period is tractable; m=144 at native 10-min
     resolution is not). Missing hours are kept as NaN — the state-space
     Kalman filter handles them natively as missing observations.
  2. Fit SARIMAX(2,1,2)(1,1,1)[24] ONCE on the raw hourly GHI training series —
     no clear-sky normalisation or derived index. The model target is the same
     physical quantity (GHI, W/m²) that the persistence baseline and the
     deep-learning models predict.
  3. Rolling-origin forecasts: with parameters FIXED at their training-set
     estimates, the Kalman filter state is advanced observation-by-observation
     through the validation and test periods (statsmodels `.extend()`, no
     re-estimation). At every on-the-hour test forecast origin t_label, a
     genuine h-step-ahead forecast of t_label + h is issued from the filtered
     state that includes all observations up to and including t_label. This
     mirrors what persistence and the deep-learning models are allowed to see:
     the past up to the forecast origin, never the future.
  4. Forecasts are clipped at zero and evaluated with the same metrics on the
     on-the-hour subset of the test manifest (rmse_day, mae_day, skill vs.
     persistence — persistence recomputed on the identical evaluated subset,
     so the skill score is internally consistent).

Note on temporal resolution:
  The manifests are at 10-min resolution but SARIMA operates hourly (m=24 is
  standard in solar forecasting literature; m=144 is intractable on 2+ years
  of data). Only test rows where t_label falls on an exact hour are evaluated,
  and the persistence reference is recomputed on that same on-the-hour subset
  so RMSE/skill are computed on a like-for-like basis (these numbers are NOT
  directly comparable to the full-resolution persistence/DL metrics reported
  elsewhere, which are computed over the full 10-min test manifest). A further
  caveat: SARIMA is fit on (and therefore forecasts) HOURLY-MEAN GHI, while
  the manifest target y is the instantaneous 10-min GHI at t_target; the
  within-hour averaging slightly smooths the target SARIMA is scored against.

Order selection:
  SARIMA_ORDER / SARIMA_SEASONAL_ORDER below were re-selected for the raw
  hourly GHI series via `notebooks/05b_sarima_order_selection.ipynb` (the
  clear-sky-index orders from the earlier version of this script do not
  necessarily transfer to raw GHI, which has very different stationarity
  and scale properties). Re-run that notebook if the data or sites change.
  The rolling-origin evaluation below changes only HOW forecasts are issued,
  not the model or its order — the selection remains valid.

Dependencies (must be installed in .venv):
    pip install statsmodels

Usage (from project root):
    python scripts/05_sarima_baseline.py --site uniandes --hours_ahead 6
    python scripts/05_sarima_baseline.py --site elpaso   --hours_ahead 1
    python scripts/05_sarima_baseline.py --site uniandes --hours_ahead 1 3 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.metrics import metrics_from_arrays, skill_score, eval_persistence

# ---------------------------------------------------------------------------
# SARIMA order — (p,d,q)(P,D,Q)[m] — selected on raw hourly GHI, see docstring
# ---------------------------------------------------------------------------
SARIMA_ORDER          = (2, 1, 2)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)   # m=24 hours (daily seasonality)

RUNS_DIR    = PROJECT_ROOT / "runs" / "sarima"
GROUND_DIR  = PROJECT_ROOT / "data" / "ground_aligned"
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SARIMA GHI benchmark (rolling-origin)")
    p.add_argument("--site",         required=True, choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",  nargs="+", type=int, default=[1, 3, 6],
                   choices=[1, 3, 6],
                   help="Forecast horizon(s) in hours (space-separated, e.g. 1 3 6)")
    p.add_argument("--day_threshold", type=float, default=20.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# SARIMA fit (training window only — parameters are never re-estimated)
# ---------------------------------------------------------------------------

def fit_train(ghi_train: pd.Series):
    """Fit SARIMAX on the raw hourly GHI training series. Returns the results
    object whose parameter estimates are reused (fixed) for all rolling
    forecasts."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            ghi_train,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
    return result


# ---------------------------------------------------------------------------
# Rolling-origin forecasting
# ---------------------------------------------------------------------------

def rolling_forecasts(
    result_train,
    ghi_hourly: pd.Series,
    origins: pd.DatetimeIndex,
    max_h: int,
) -> dict[pd.Timestamp, pd.Series]:
    """h-step-ahead forecasts from each origin, with parameters fixed at their
    training estimates and the Kalman filter state advanced one hourly
    observation at a time (statsmodels ``.extend()`` — no re-estimation).

    Args:
        result_train : fitted SARIMAXResults on the training slice.
        ghi_hourly   : full regular hourly series (NaN = missing observation),
                       covering train start through at least the last origin.
        origins      : sorted on-the-hour forecast origins (test t_label values).
                       The forecast issued at origin t uses observations up to
                       and INCLUDING t, mirroring persistence/DL information.
        max_h        : longest horizon needed, in hours.

    Returns:
        {origin: forecast Series of length max_h indexed t+1h … t+max_h,
         clipped at 0}.
    """
    origins = pd.DatetimeIndex(origins).sort_values()
    first_origin, last_origin = origins[0], origins[-1]
    origin_set = set(origins)

    train_end = result_train.model.data.dates[-1]

    # Advance the filter through the gap (validation period etc.) in one call.
    res = result_train
    gap = ghi_hourly.loc[train_end + pd.Timedelta(hours=1): first_origin - pd.Timedelta(hours=1)]
    if len(gap) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = res.extend(gap)

    fc_by_origin: dict[pd.Timestamp, pd.Series] = {}
    t0 = time.time()
    hours = ghi_hourly.loc[first_origin:last_origin].index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(hours):
            # Extend state with the observation at t (NaN handled as missing).
            res = res.extend(ghi_hourly.loc[[t]])
            if t in origin_set:
                fc = res.forecast(steps=max_h)
                if not isinstance(fc.index, pd.DatetimeIndex):
                    fc.index = pd.date_range(
                        t + pd.Timedelta(hours=1), periods=max_h, freq="1h", tz=t.tz
                    )
                fc_by_origin[t] = fc.clip(lower=0.0)
            if (i + 1) % 500 == 0:
                print(f"    rolling: {i+1}/{len(hours)} hours "
                      f"({len(fc_by_origin)} origins done, {time.time()-t0:.0f}s)",
                      flush=True)

    print(f"  Rolling forecasts: {len(fc_by_origin)} origins in {time.time()-t0:.0f}s")
    return fc_by_origin


# ---------------------------------------------------------------------------
# Evaluate one horizon
# ---------------------------------------------------------------------------

def evaluate_horizon(
    site: str,
    hours_ahead: int,
    ground: pd.DataFrame,
    fc_by_origin: dict[pd.Timestamp, pd.Series],
    day_threshold: float,
) -> dict:
    """
    Load test manifest for (site, hours_ahead), filter to hourly t_label rows,
    look up the h-step-ahead SARIMA prediction issued at each origin, compute
    metrics.
    """
    manifest_dir = DATASET_DIR / site / f"h{hours_ahead}"
    test_man  = pd.read_parquet(manifest_dir / "manifest_test.parquet")
    train_man = pd.read_parquet(manifest_dir / "manifest_train.parquet")
    val_man   = pd.read_parquet(manifest_dir / "manifest_val.parquet")

    with open(manifest_dir / "dataset_meta.json") as f:
        meta = json.load(f)

    # Filter test manifest to on-the-hour t_label rows
    t_label = pd.to_datetime(test_man["t_label"], utc=True)
    on_hour = t_label.dt.minute == 0
    test_sub = test_man[on_hour.values].copy()
    t_label_h  = pd.to_datetime(test_sub["t_label"],  utc=True)
    t_target_h = pd.to_datetime(test_sub["t_target"], utc=True)
    y_true = test_sub["y"].astype(float).to_numpy()

    if len(test_sub) == 0:
        raise ValueError(f"No on-the-hour test rows found for {site} h{hours_ahead}")

    # h-step-ahead prediction: the forecast issued at origin t_label, evaluated
    # at t_target = t_label + h. Same target variable (GHI, W/m²) and same `y`
    # ground-truth column as every other model.
    y_pred = np.full(len(test_sub), np.nan)
    for i, (t_org, t_tgt) in enumerate(zip(t_label_h, t_target_h)):
        fc = fc_by_origin.get(t_org)
        if fc is not None and t_tgt in fc.index:
            y_pred[i] = float(fc.loc[t_tgt])

    valid = np.isfinite(y_pred)
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  [warn] {n_missing} test targets without a rolling forecast")

    test_metrics = metrics_from_arrays(y_true_v, y_pred_v, day_threshold)

    # Persistence baseline recomputed on the SAME evaluated on-the-hour subset,
    # so the skill score is internally consistent (see module docstring for
    # why this differs from the full-resolution persistence reported for DL
    # models).
    pers = eval_persistence(test_sub[valid], ground, day_threshold)
    test_metrics["skill_vs_persistence"]     = skill_score(test_metrics["rmse"],     pers["rmse"])
    test_metrics["skill_day_vs_persistence"] = skill_score(test_metrics["rmse_day"], pers["rmse_day"])

    # Val/train persistence for reference (full manifests — informational only)
    pers_val   = eval_persistence(val_man,   ground, day_threshold)
    pers_train = eval_persistence(train_man, ground, day_threshold)

    print(
        f"  h{hours_ahead}h | n={test_metrics['n']} (of {len(test_man)} total, "
        f"{on_hour.sum()} on-hour) | "
        f"RMSE={test_metrics['rmse']:.1f}  RMSE_day={test_metrics['rmse_day']:.1f}  "
        f"skill_day={test_metrics['skill_day_vs_persistence']:.3f}"
    )

    return {
        "hours_ahead":     hours_ahead,
        "horizon_steps":   int(meta["horizon_steps"]),
        "test_metrics":    test_metrics,
        "persistence_train": pers_train,
        "persistence_val":   pers_val,
        "persistence_test":  pers,
        "n_on_hour_evaluated": int(valid.sum()),
        "n_total_test":        int(len(test_man)),
        "n_on_hour_available": int(on_hour.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    site = args.site

    print(f"SARIMA baseline (rolling-origin) | site={site} | horizons={args.hours_ahead}h")

    # ------------------------------------------------------------------
    # Load ground data — same tabular GHI table used by every other model
    # (10-min, UTC-indexed)
    # ------------------------------------------------------------------
    ground_path = GROUND_DIR / f"ground_10min_utc_{site}.parquet"
    assert ground_path.exists(), f"Missing: {ground_path}"
    ground = pd.read_parquet(ground_path)
    assert "ghi" in ground.columns and str(ground.index.tz) == "UTC"

    # ------------------------------------------------------------------
    # Resample to hourly on a REGULAR index — fit directly on raw GHI (W/m²),
    # no clear-sky normalisation. Missing hours stay NaN: the state-space
    # Kalman filter treats NaN as a missing observation, which keeps the index
    # regular so that .extend() advances exactly one hour per observation.
    # ------------------------------------------------------------------
    ghi_hourly = ground["ghi"].resample("1h").mean()  # NaN where no data

    # ------------------------------------------------------------------
    # Training period (union of train splits across all horizons)
    # ------------------------------------------------------------------
    train_starts, train_ends = [], []
    for h in args.hours_ahead:
        man_dir = DATASET_DIR / site / f"h{h}"
        train_man = pd.read_parquet(man_dir / "manifest_train.parquet")
        t_labels = pd.to_datetime(train_man["t_label"], utc=True)
        train_starts.append(t_labels.min())
        train_ends.append(t_labels.max())

    train_start = min(train_starts).floor("1h")
    train_end   = max(train_ends).floor("1h")
    print(f"Training window: {train_start.date()} → {train_end.date()}")

    ghi_train = ghi_hourly.loc[train_start:train_end].copy()
    print(f"Training GHI series: {len(ghi_train):,} hourly slots "
          f"({ghi_train.isna().sum():,} missing, kept as NaN; "
          f"mean={ghi_train.mean():.1f}, std={ghi_train.std():.1f} W/m²)")

    # ------------------------------------------------------------------
    # Fit once on train
    # ------------------------------------------------------------------
    print(f"Fitting SARIMAX{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} on raw hourly GHI ...",
          end=" ", flush=True)
    t0 = time.time()
    result_train = fit_train(ghi_train)
    print(f"{time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Collect forecast origins: union of on-the-hour test t_labels over all
    # requested horizons; forecast max(horizons) steps from each origin.
    # ------------------------------------------------------------------
    origin_set: set[pd.Timestamp] = set()
    for h in args.hours_ahead:
        man_dir = DATASET_DIR / site / f"h{h}"
        test_man = pd.read_parquet(man_dir / "manifest_test.parquet")
        t_labels = pd.to_datetime(test_man["t_label"], utc=True)
        origin_set.update(t_labels[t_labels.dt.minute == 0])
    origins = pd.DatetimeIndex(sorted(origin_set))
    max_h = max(args.hours_ahead)
    print(f"Forecast origins: {len(origins):,} on-the-hour test t_labels "
          f"({origins.min()} → {origins.max()}), max horizon {max_h}h")

    fc_by_origin = rolling_forecasts(result_train, ghi_hourly, origins, max_h)

    # ------------------------------------------------------------------
    # Evaluate each horizon
    # ------------------------------------------------------------------
    print("\n=== Test evaluation ===")
    results_by_horizon = {}
    for h in args.hours_ahead:
        res = evaluate_horizon(
            site=site,
            hours_ahead=h,
            ground=ground,
            fc_by_origin=fc_by_origin,
            day_threshold=args.day_threshold,
        )
        results_by_horizon[h] = res

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts   = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    run_name = f"{site}_SARIMA_{'_'.join(f'h{h}' for h in args.hours_ahead)}_{run_ts}"
    run_dir  = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name":   run_name,
        "site":       site,
        "arch":       "SARIMA",
        "model": {
            "order":          list(SARIMA_ORDER),
            "seasonal_order": list(SARIMA_SEASONAL_ORDER),
            "resolution":     "hourly",
            "target":         "ghi_raw_wm2",
            "evaluation":     "rolling_origin_fixed_params",
            "n_origins":      len(fc_by_origin),
        },
        "results":  results_by_horizon,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
