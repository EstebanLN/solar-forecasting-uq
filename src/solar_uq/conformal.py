"""
Conformal Prediction utilities for GHI point forecasting.

Currently implements Split (Inductive) CP with an absolute-residual
nonconformity score.  The finite-sample correction
    q_level = ceil((n+1)(1-alpha)) / n
guarantees exact marginal coverage ≥ 1-alpha when calibration and test
samples are exchangeable.

Usage
-----
    from solar_uq.conformal import SplitCP, evaluate_coverage_by_alpha

    cp = SplitCP().calibrate(y_val, yhat_val, alpha=0.10)
    lo, hi = cp.predict(yhat_test)
    result = cp.evaluate(y_test, yhat_test)

    # Sweep over multiple alpha levels in one call:
    rows = evaluate_coverage_by_alpha(y_val, yhat_val, y_test, yhat_test)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SplitCP:
    """
    Split (Inductive) Conformal Prediction — symmetric, fixed-width intervals.

    Nonconformity score:  s_i = |y_i - ŷ_i|  (absolute residual, W/m²).
    Prediction interval:  ŷ ± q_hat
    """

    q_hat: float = 0.0
    alpha: float = 0.1
    n_cal: int = 0

    def calibrate(
        self,
        y_cal: np.ndarray,
        yhat_cal: np.ndarray,
        alpha: float = 0.1,
    ) -> "SplitCP":
        """Compute the conformal quantile from a calibration set.

        Parameters
        ----------
        y_cal, yhat_cal : physical-unit (W/m²) arrays of the same length.
        alpha           : desired miscoverage rate (e.g. 0.10 → 90% intervals).
        """
        scores = np.abs(y_cal - yhat_cal)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.alpha = alpha
        self.n_cal = n
        return self

    def predict(self, yhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) interval bounds for each prediction."""
        return yhat - self.q_hat, yhat + self.q_hat

    def evaluate(
        self,
        y_true: np.ndarray,
        yhat: np.ndarray,
        day_threshold: float = 20.0,
    ) -> Dict[str, float]:
        """Compute coverage and interval-width metrics on a test set.

        Returns a dict with overall and daytime-only metrics.
        """
        lo, hi  = self.predict(yhat)
        covered = (y_true >= lo) & (y_true <= hi)
        width   = hi - lo
        day     = y_true >= day_threshold

        result: Dict = {
            "alpha":              self.alpha,
            "target_coverage":    round(1 - self.alpha, 4),
            "q_hat_wm2":          round(self.q_hat, 2),
            "n_cal":              self.n_cal,
            "n_test":             int(len(y_true)),
            "coverage":           round(float(covered.mean()), 4),
            "mean_width_wm2":     round(float(width.mean()), 2),
            "n_day":              int(day.sum()),
            "coverage_day":       round(float(covered[day].mean()), 4) if day.any() else None,
            "mean_width_day_wm2": round(float(width[day].mean()), 2)   if day.any() else None,
        }
        return result


def evaluate_coverage_by_alpha(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    alphas: Optional[List[float]] = None,
    day_threshold: float = 20.0,
) -> List[Dict]:
    """Calibrate and evaluate SplitCP for multiple alpha levels.

    Returns a list of result dicts (one per alpha), suitable for
    serialisation or pandas DataFrame construction.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.20]
    return [
        SplitCP().calibrate(y_cal, yhat_cal, alpha=a).evaluate(
            y_test, yhat_test, day_threshold=day_threshold
        )
        for a in alphas
    ]
