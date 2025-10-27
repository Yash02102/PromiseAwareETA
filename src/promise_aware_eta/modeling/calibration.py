"""Calibration utilities for conformalized quantile regression."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert an input array-like object to a 1-D numpy array."""

    if isinstance(array, pd.Series):
        return array.to_numpy(dtype=float, copy=False)
    return np.asarray(array, dtype=float)


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile with finite-sample correction."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if scores.size == 0:
        raise ValueError("scores array must not be empty")

    n = scores.size
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = max(1, min(rank, n))
    partitioned = np.partition(scores, rank - 1)
    return float(partitioned[rank - 1])


def _compute_conformal_scores(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return split-conformal (CQR) nonconformity scores."""

    if y_true.shape != lower.shape or y_true.shape != upper.shape:
        raise ValueError("y_true, lower, and upper must have matching shapes")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")

    scores = np.maximum(lower - y_true, y_true - upper)
    return np.maximum(scores, 0.0)


@dataclass
class CoverageDiagnostics:
    """Summary statistics describing interval coverage performance."""

    coverage: float
    target_coverage: float
    miscoverage_error: float
    lower_violation_rate: float
    upper_violation_rate: float
    mean_interval_width: float

    def as_dict(self) -> Mapping[str, float]:
        """Return the diagnostics as a plain dictionary."""

        return asdict(self)


def compute_coverage_diagnostics(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    target_coverage: float,
) -> CoverageDiagnostics:
    """Compute empirical coverage diagnostics for prediction intervals."""

    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be in (0, 1)")

    y_arr = _to_numpy(y_true)
    lower_arr = _to_numpy(lower)
    upper_arr = _to_numpy(upper)

    if y_arr.shape != lower_arr.shape or y_arr.shape != upper_arr.shape:
        raise ValueError("y_true, lower, and upper must have matching shapes")

    covered = (y_arr >= lower_arr) & (y_arr <= upper_arr)
    coverage = float(np.mean(covered))
    lower_violation = float(np.mean(y_arr < lower_arr))
    upper_violation = float(np.mean(y_arr > upper_arr))
    width = float(np.mean(upper_arr - lower_arr))

    return CoverageDiagnostics(
        coverage=coverage,
        target_coverage=target_coverage,
        miscoverage_error=coverage - target_coverage,
        lower_violation_rate=lower_violation,
        upper_violation_rate=upper_violation,
        mean_interval_width=width,
    )


class SplitConformalQuantileCalibrator:
    """Split-conformal calibrator for quantile regression intervals."""

    def __init__(self, alpha: float):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.conformal_quantile_: float | None = None
        self.calibration_diagnostics_: CoverageDiagnostics | None = None

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if the calibrator has been fitted."""

        return self.conformal_quantile_ is not None

    def fit(
        self,
        y_calib: ArrayLike,
        *,
        lower_pred: ArrayLike,
        upper_pred: ArrayLike,
    ) -> "SplitConformalQuantileCalibrator":
        """Estimate the conformal adjustment from calibration residuals."""

        y_arr = _to_numpy(y_calib)
        lower_arr = _to_numpy(lower_pred)
        upper_arr = _to_numpy(upper_pred)

        scores = _compute_conformal_scores(y_arr, lower_arr, upper_arr)
        quantile = _finite_sample_quantile(scores, self.alpha)
        self.conformal_quantile_ = quantile

        calibrated_lower = lower_arr - quantile
        calibrated_upper = upper_arr + quantile
        self.calibration_diagnostics_ = compute_coverage_diagnostics(
            y_arr,
            calibrated_lower,
            calibrated_upper,
            target_coverage=1 - self.alpha,
        )
        return self

    def predict(
        self,
        *,
        lower_pred: ArrayLike,
        upper_pred: ArrayLike,
    ) -> pd.DataFrame:
        """Apply the learned conformal adjustment to prediction bounds."""

        if self.conformal_quantile_ is None:
            raise RuntimeError("Calibrator must be fitted before calling predict().")

        lower_arr = _to_numpy(lower_pred)
        upper_arr = _to_numpy(upper_pred)

        if lower_arr.shape != upper_arr.shape:
            raise ValueError("lower_pred and upper_pred must have matching shapes")

        adjusted_lower = lower_arr - self.conformal_quantile_
        adjusted_upper = upper_arr + self.conformal_quantile_

        return pd.DataFrame({"lower": adjusted_lower, "upper": adjusted_upper})

    def calibrate(
        self,
        y_calib: ArrayLike,
        *,
        lower_calib: ArrayLike,
        upper_calib: ArrayLike,
        lower_pred: ArrayLike,
        upper_pred: ArrayLike,
    ) -> pd.DataFrame:
        """Convenience method combining ``fit`` and ``predict``."""

        self.fit(y_calib, lower_pred=lower_calib, upper_pred=upper_calib)
        return self.predict(lower_pred=lower_pred, upper_pred=upper_pred)

