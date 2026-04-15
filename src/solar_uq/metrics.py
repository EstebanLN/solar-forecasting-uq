"""Evaluation metrics for GHI point forecasting."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    day_threshold: float = 20.0,
) -> Dict[str, float]:
    """RMSE, MAE (all samples) and RMSE_day, MAE_day (y_true >= day_threshold)."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    day_mask = y_true >= day_threshold
    if day_mask.sum() > 0:
        rmse_day = float(np.sqrt(np.mean((y_true[day_mask] - y_pred[day_mask]) ** 2)))
        mae_day  = float(np.mean(np.abs(y_true[day_mask] - y_pred[day_mask])))
        n_day = int(day_mask.sum())
    else:
        rmse_day, mae_day, n_day = float("nan"), float("nan"), 0

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "n_day": n_day,
        "rmse_day": rmse_day,
        "mae_day": mae_day,
        "day_threshold": float(day_threshold),
    }


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """Skill score vs a baseline: 1 - rmse_model / rmse_baseline."""
    return float(1.0 - rmse_model / (rmse_baseline + 1e-12))


def eval_persistence(
    manifest: pd.DataFrame,
    ground_df: pd.DataFrame,
    day_threshold: float = 20.0,
) -> Dict[str, float]:
    """Persistence forecast: ŷ(t+H) = GHI(t), evaluated against manifest y."""
    t_label = pd.to_datetime(manifest["t_label"], utc=True)
    y_true  = manifest["y"].astype(float).to_numpy()
    y_hat   = ground_df.reindex(t_label)["ghi"].to_numpy()
    return metrics_from_arrays(y_true, y_hat, day_threshold=day_threshold)
