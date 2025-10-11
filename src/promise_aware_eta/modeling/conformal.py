"""Conformal calibration utilities for quantile prediction intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class RollingConformalDiagnostics:
    """Diagnostics captured when applying rolling split conformal calibration."""

    lower_quantile: float
    upper_quantile: float
    target_coverage: float
    num_splits: int
    adjustments: List[float]
    split_coverages: List[float]
    split_widths: List[float]


def _nonconformity_scores(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Compute two-sided nonconformity scores for prediction intervals."""

    return np.maximum(lower - y_true, y_true - upper)


def _split_conformal_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """Return the conformal adjustment quantile for the desired coverage."""

    if scores.size == 0:
        return 0.0
    alpha = 1.0 - float(target_coverage)
    # Conformal quantile following the ceil((n + 1) * (1 - alpha)) / n rule
    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    # Use partition to avoid full sort for efficiency
    return float(np.partition(scores, rank - 1)[rank - 1])


def rolling_split_conformal(
    *,
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    target_coverage: float,
    lower_quantile: float,
    upper_quantile: float,
    num_splits: int = 0,
) -> Tuple[np.ndarray, np.ndarray, RollingConformalDiagnostics]:
    """Apply rolling split conformal calibration to prediction intervals.

    Parameters
    ----------
    y_true:
        Observed outcomes corresponding to the prediction interval outputs.
    lower / upper:
        Arrays containing the lower and upper bounds of the prediction interval.
    target_coverage:
        Desired marginal coverage for the interval (e.g., 0.9 for a 90% interval).
    lower_quantile / upper_quantile:
        Quantiles that produced the lower and upper bounds, useful for logging.
    num_splits:
        Number of sequential splits over which to apply rolling calibration. When
        ``num_splits`` is 0 or 1 the entire validation window is treated as a
        single segment and no cross-split calibration is applied beyond a global
        adjustment computed on the full sample.

    Returns
    -------
    adjusted_lower, adjusted_upper, diagnostics
        Calibrated interval bounds and bookkeeping information describing the
        adjustments applied across splits.
    """

    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if y_true.shape[0] != lower.shape[0] or lower.shape[0] != upper.shape[0]:
        raise ValueError("All input arrays must share the same length.")

    n = y_true.shape[0]
    if n == 0:
        diagnostics = RollingConformalDiagnostics(
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
            target_coverage=float(target_coverage),
            num_splits=0,
            adjustments=[],
            split_coverages=[],
            split_widths=[],
        )
        return lower, upper, diagnostics

    if num_splits and num_splits > 0:
        splits = np.array_split(np.arange(n), num_splits)
    else:
        splits = [np.arange(n)]

    adjusted_lower = lower.copy()
    adjusted_upper = upper.copy()

    calibration_indices: np.ndarray = np.array([], dtype=int)
    adjustments: List[float] = []
    coverages: List[float] = []
    widths: List[float] = []

    for indices in splits:
        if indices.size == 0:
            adjustments.append(0.0)
            coverages.append(float("nan"))
            widths.append(float("nan"))
            continue

        if calibration_indices.size == 0:
            adjustment = 0.0
        else:
            scores = _nonconformity_scores(
                y_true[calibration_indices],
                lower[calibration_indices],
                upper[calibration_indices],
            )
            adjustment = _split_conformal_quantile(scores, target_coverage)

        adjusted_lower_segment = lower[indices] - adjustment
        adjusted_upper_segment = upper[indices] + adjustment
        adjusted_lower[indices] = adjusted_lower_segment
        adjusted_upper[indices] = adjusted_upper_segment

        coverage = float(
            np.mean(
                (y_true[indices] >= adjusted_lower_segment)
                & (y_true[indices] <= adjusted_upper_segment)
            )
        )
        width = float(np.mean(adjusted_upper_segment - adjusted_lower_segment))

        adjustments.append(float(adjustment))
        coverages.append(coverage)
        widths.append(width)

        calibration_indices = np.concatenate([calibration_indices, indices])

    diagnostics = RollingConformalDiagnostics(
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        target_coverage=float(target_coverage),
        num_splits=len(splits),
        adjustments=adjustments,
        split_coverages=coverages,
        split_widths=widths,
    )
    return adjusted_lower, adjusted_upper, diagnostics
