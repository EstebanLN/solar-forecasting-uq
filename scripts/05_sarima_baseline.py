#!/usr/bin/env python
"""
05_sarima_baseline.py — SARIMA statistical benchmark for GHI forecasting.

Methodology:
  1. Compute clear-sky GHI with pvlib (Ineichen model) at hourly resolution.
  2. Derive clear-sky index  k = GHI / GHI_cs  (0 at night, ~[0,1.2] during day).
  3. Fit SARIMAX(2,1,2)(1,1,1)[24] on training k series (hourly, m=24).
  4. Forecast k at each test target timestamp; convert back: GHI_pred = k_pred * GHI_cs.
  5. Evaluate with same metrics as DL baselines (rmse_day, mae_day, skill vs persistence).

Note on temporal resolution:
  The manifests are at 10-min resolution but SARIMA operates hourly (m=24 is standard
  in solar forecasting literature; m=144 is intractable on 2+ years of data).
  Only test rows where t_label falls on an exact hour are evaluated.

Dependencies (must be installed in .venv):
    pip install pvlib statsmodels

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
# Site coordinates  — verify before running
# ---------------------------------------------------------------------------
SITE_META: dict[str, dict] = {
    "uniandes": {
        "lat":       4.6024,
        "lon":      -74.0663,
        "elevation": 2600,    # m a.s.l., Bogotá
        "tz":       "America/Bogota",
    },
    "elpaso": {
        # El Paso, César — centro del municipio (rango confirmado por usuario)
        "lat":       9.737,
        "lon":      -73.695,
        "elevation": 50,
        "tz":       "America/Bogota",
    },
}

# SARIMA order — (p,d,q)(P,D,Q)[m]
SARIMA_ORDER          = (2, 1, 2)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)   # m=24 hours (daily seasonality)

CS_FLOOR   = 10.0   # W/m²: below this, k is set to 0 (night/twilight)
K_MAX      = 1.5    # hard ceiling on clear-sky index
RUNS_DIR   = PROJECT_ROOT / "runs" / "sarima"

GROUND_DIR = PROJECT_ROOT / "data" / "ground_aligned"
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SARIMA GHI benchmark")
    p.add_argument("--site",         required=True, choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",  nargs="+", type=int, default=[1, 3, 6],
                   choices=[1, 3, 6],
                   help="Forecast horizon(s) in hours (space-separated, e.g. 1 3 6)")
    p.add_argument("--day_threshold", type=float, default=20.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clear-sky index series
# ---------------------------------------------------------------------------

def compute_clearsky_index(
    ghi_hourly: pd.Series,
    lat: float,
    lon: float,
    elevation: float,
    tz: str,
) -> pd.Series:
    """Return hourly clear-sky index series aligned to ghi_hourly.index."""
    try:
        import pvlib
    except ImportError:
        raise ImportError("pvlib is required: pip install pvlib")

    loc = pvlib.location.Location(
        latitude=lat, longitude=lon, altitude=elevation, tz=tz
    )
    times_local = ghi_hourly.index.tz_convert(tz)
    cs = loc.get_clearsky(times_local, model="ineichen")    # W/m²
    ghi_cs = cs["ghi"].values.astype(np.float64)

    ghi_vals = ghi_hourly.values.astype(np.float64)
    k = np.where(ghi_cs >= CS_FLOOR, ghi_vals / ghi_cs, 0.0)
    k = np.clip(k, 0.0, K_MAX)
    k = np.nan_to_num(k, nan=0.0)
    return pd.Series(k, index=ghi_hourly.index, name="k")


# ---------------------------------------------------------------------------
# SARIMA fit + forecast
# ---------------------------------------------------------------------------

def fit_and_forecast(
    k_train: pd.Series,
    n_forecast: int,
) -> pd.Series:
    """Fit SARIMAX on k_train; return out-of-sample forecast of length n_forecast."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            k_train,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)

    fc = result.forecast(steps=n_forecast)
    # statsmodels may return RangeIndex when k_train has no inferred freq
    if not isinstance(fc.index, pd.DatetimeIndex):
        start = k_train.index[-1] + pd.Timedelta(hours=1)
        fc.index = pd.date_range(start=start, periods=len(fc), freq="1h")
    fc = fc.clip(0.0, K_MAX)
    return fc


# ---------------------------------------------------------------------------
# Evaluate one horizon
# ---------------------------------------------------------------------------

def evaluate_horizon(
    site: str,
    hours_ahead: int,
    ground: pd.DataFrame,
    k_series: pd.Series,
    ghi_cs_series: pd.Series,
    fc_series: pd.Series,
    day_threshold: float,
) -> dict:
    """
    Load test manifest for (site, hours_ahead), filter to hourly t_label rows,
    look up SARIMA predictions from fc_series, compute metrics.
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
    t_label_h = pd.to_datetime(test_sub["t_label"], utc=True)
    t_target_h = pd.to_datetime(test_sub["t_target"], utc=True)
    y_true = test_sub["y"].astype(float).to_numpy()

    if len(test_sub) == 0:
        raise ValueError(f"No on-the-hour test rows found for {site} h{hours_ahead}")

    # Look up predicted k at t_target; convert to GHI
    y_pred = np.full(len(test_sub), np.nan)
    for i, t_tgt in enumerate(t_target_h):
        if t_tgt in fc_series.index:
            k_pred = float(fc_series.loc[t_tgt])
            ghi_cs_tgt = float(ghi_cs_series.get(t_tgt, 0.0))
            y_pred[i] = k_pred * ghi_cs_tgt if ghi_cs_tgt >= CS_FLOOR else 0.0

    valid = np.isfinite(y_pred)
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  [warn] {n_missing} test targets not covered by forecast index")

    test_metrics = metrics_from_arrays(y_true_v, y_pred_v, day_threshold)

    # Persistence baseline on the same filtered subset
    pers = eval_persistence(test_sub[valid], ground, day_threshold)
    test_metrics["skill_vs_persistence"]     = skill_score(test_metrics["rmse"],     pers["rmse"])
    test_metrics["skill_day_vs_persistence"] = skill_score(test_metrics["rmse_day"], pers["rmse_day"])

    # Val/train persistence for reference
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
    smeta = SITE_META[site]

    if smeta["lat"] is None:
        raise ValueError(
            f"Coordinates for site '{site}' are not set. "
            f"Edit SITE_META in this script and fill in lat/lon/elevation."
        )

    print(f"SARIMA baseline | site={site} | horizons={args.hours_ahead}h")
    print(f"Location: lat={smeta['lat']}, lon={smeta['lon']}, elev={smeta['elevation']}m")

    # ------------------------------------------------------------------
    # Load ground data (10-min, UTC-indexed)
    # ------------------------------------------------------------------
    ground_path = GROUND_DIR / f"ground_10min_utc_{site}.parquet"
    assert ground_path.exists(), f"Missing: {ground_path}"
    ground = pd.read_parquet(ground_path)
    assert "ghi" in ground.columns and str(ground.index.tz) == "UTC"

    # ------------------------------------------------------------------
    # Resample to hourly
    # ------------------------------------------------------------------
    ghi_hourly = ground["ghi"].resample("1h").mean()
    ghi_hourly = ghi_hourly.dropna()

    # ------------------------------------------------------------------
    # Clear-sky GHI and clear-sky index
    # ------------------------------------------------------------------
    print("Computing clear-sky GHI ...", end=" ", flush=True)
    t0 = time.time()
    k_all = compute_clearsky_index(
        ghi_hourly,
        lat=smeta["lat"], lon=smeta["lon"],
        elevation=smeta["elevation"], tz=smeta["tz"],
    )
    # Store GHI_cs for back-conversion at prediction time
    import pvlib
    loc = pvlib.location.Location(
        latitude=smeta["lat"], longitude=smeta["lon"],
        altitude=smeta["elevation"], tz=smeta["tz"],
    )
    times_local = ghi_hourly.index.tz_convert(smeta["tz"])
    cs = loc.get_clearsky(times_local, model="ineichen")
    ghi_cs_hourly = pd.Series(
        cs["ghi"].values.astype(np.float64),
        index=ghi_hourly.index,
        name="ghi_cs",
    )
    print(f"{time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Determine training period (union of train splits across all horizons)
    # Using the earliest start and latest end of train manifests
    # ------------------------------------------------------------------
    train_starts, train_ends = [], []
    for h in args.hours_ahead:
        man_dir = DATASET_DIR / site / f"h{h}"
        train_man = pd.read_parquet(man_dir / "manifest_train.parquet")
        t_labels = pd.to_datetime(train_man["t_label"], utc=True)
        train_starts.append(t_labels.min())
        train_ends.append(t_labels.max())

    train_start = min(train_starts)
    train_end   = max(train_ends)
    print(f"Training window: {train_start.date()} → {train_end.date()}")

    k_train = k_all.loc[
        (k_all.index >= train_start) & (k_all.index <= train_end)
    ].copy()
    print(f"Training k series: {len(k_train):,} hourly observations")

    # ------------------------------------------------------------------
    # Fit SARIMA
    # ------------------------------------------------------------------
    print(f"Fitting SARIMAX{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} ...", end=" ", flush=True)
    t0 = time.time()

    # Forecast through the end of the latest test period
    test_ends = []
    for h in args.hours_ahead:
        man_dir = DATASET_DIR / site / f"h{h}"
        test_man = pd.read_parquet(man_dir / "manifest_test.parquet")
        t_targets = pd.to_datetime(test_man["t_target"], utc=True)
        test_ends.append(t_targets.max())
    test_end_global = max(test_ends)

    n_forecast = int((test_end_global - train_end).total_seconds() / 3600) + 24
    fc_series = fit_and_forecast(k_train, n_forecast)
    print(f"{time.time()-t0:.1f}s  |  forecast steps={len(fc_series)}")

    # Align forecast index to UTC
    if fc_series.index.tz is None:
        fc_series.index = fc_series.index.tz_localize("UTC")
    else:
        fc_series.index = fc_series.index.tz_convert("UTC")

    ghi_cs_hourly_utc = ghi_cs_hourly.copy()
    if ghi_cs_hourly_utc.index.tz is None:
        ghi_cs_hourly_utc.index = ghi_cs_hourly_utc.index.tz_localize("UTC")
    else:
        ghi_cs_hourly_utc.index = ghi_cs_hourly_utc.index.tz_convert("UTC")

    # Extend ghi_cs to cover forecast period
    forecast_times = fc_series.index
    new_times = forecast_times.difference(ghi_cs_hourly_utc.index)
    if len(new_times) > 0:
        new_times_local = new_times.tz_convert(smeta["tz"])
        cs_new = loc.get_clearsky(new_times_local, model="ineichen")
        extra = pd.Series(
            cs_new["ghi"].values.astype(np.float64),
            index=new_times,
            name="ghi_cs",
        )
        ghi_cs_hourly_utc = pd.concat([ghi_cs_hourly_utc, extra]).sort_index()

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
            k_series=k_all,
            ghi_cs_series=ghi_cs_hourly_utc,
            fc_series=fc_series,
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
            "cs_model":       "ineichen (pvlib)",
            "cs_floor_wm2":   CS_FLOOR,
        },
        "location": {k: smeta[k] for k in ("lat", "lon", "elevation", "tz")},
        "results":  results_by_horizon,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
