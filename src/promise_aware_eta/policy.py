"""Delivery promise policy simulation, evaluation, and fairness utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series
CostFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert an array-like input into a one-dimensional ``numpy`` array."""

    if isinstance(array, pd.Series):
        return array.to_numpy(dtype=float, copy=False)
    return np.asarray(array, dtype=float)


def _validate_lengths(*arrays: np.ndarray) -> None:
    """Ensure that all provided arrays share the same length."""

    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("All input arrays must have matching lengths.")


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Return the (optionally) weighted mean of ``values``."""

    if weights is None:
        return float(np.mean(values))
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive.")
    return float(np.dot(values, weights) / total_weight)


def asymmetric_cost(
    actual: ArrayLike,
    promised: ArrayLike,
    *,
    late_penalty: float = 5.0,
    early_penalty: float = 1.0,
) -> np.ndarray:
    """Default asymmetric cost that penalizes late deliveries more heavily."""

    if late_penalty < 0 or early_penalty < 0:
        raise ValueError("Penalties must be non-negative.")

    actual_arr = _to_numpy(actual)
    promised_arr = _to_numpy(promised)
    _validate_lengths(actual_arr, promised_arr)

    diff = actual_arr - promised_arr
    late_cost = np.where(diff > 0, diff * late_penalty, 0.0)
    early_cost = np.where(diff < 0, -diff * early_penalty, 0.0)
    return late_cost + early_cost


@dataclass(frozen=True)
class PromisePolicy:
    """Representation of a promise policy based on a quantile forecast."""

    quantile: float
    cost_fn: CostFunction
    name: str | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if not 0 < self.quantile < 1:
            raise ValueError("Policy quantile must lie in (0, 1).")

    @property
    def display_name(self) -> str:
        """Return a human-friendly identifier for reporting."""

        return self.name or f"q{int(round(self.quantile * 100))}"


@dataclass
class PromiseEvaluation:
    """Summary statistics for a policy applied to a dataset."""

    policy: PromisePolicy
    coverage: float
    average_promised: float
    average_actual: float
    average_lateness: float
    average_earliness: float
    expected_cost: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a dictionary representation for downstream logging."""

        return {
            "policy": self.policy.display_name,
            "quantile": self.policy.quantile,
            "coverage": self.coverage,
            "average_promised": self.average_promised,
            "average_actual": self.average_actual,
            "average_lateness": self.average_lateness,
            "average_earliness": self.average_earliness,
            "expected_cost": self.expected_cost,
        }


def evaluate_policy(
    policy: PromisePolicy,
    *,
    actual: ArrayLike,
    quantile_predictions: Mapping[float, ArrayLike],
    weights: ArrayLike | None = None,
) -> PromiseEvaluation:
    """Evaluate a single promise policy against realized delivery outcomes."""

    try:
        promised = quantile_predictions[policy.quantile]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(
            f"Quantile {policy.quantile} not available in predictions."
        ) from exc

    actual_arr = _to_numpy(actual)
    promised_arr = _to_numpy(promised)
    _validate_lengths(actual_arr, promised_arr)

    weight_arr: np.ndarray | None = None
    if weights is not None:
        weight_arr = _to_numpy(weights)
        _validate_lengths(actual_arr, weight_arr)

    lateness = np.clip(actual_arr - promised_arr, a_min=0.0, a_max=None)
    earliness = np.clip(promised_arr - actual_arr, a_min=0.0, a_max=None)
    on_time = (lateness == 0).astype(float)
    cost = policy.cost_fn(actual_arr, promised_arr)

    coverage = _weighted_mean(on_time, weight_arr)
    avg_promised = _weighted_mean(promised_arr, weight_arr)
    avg_actual = _weighted_mean(actual_arr, weight_arr)
    avg_late = _weighted_mean(lateness, weight_arr)
    avg_early = _weighted_mean(earliness, weight_arr)
    expected_cost = _weighted_mean(cost, weight_arr)

    return PromiseEvaluation(
        policy=policy,
        coverage=coverage,
        average_promised=avg_promised,
        average_actual=avg_actual,
        average_lateness=avg_late,
        average_earliness=avg_early,
        expected_cost=expected_cost,
    )


def evaluate_policies(
    policies: Iterable[PromisePolicy],
    *,
    actual: ArrayLike,
    quantile_predictions: Mapping[float, ArrayLike],
    weights: ArrayLike | None = None,
) -> pd.DataFrame:
    """Evaluate multiple policies and return a tidy ``DataFrame`` summary."""

    evaluations = [
        evaluate_policy(policy, actual=actual, quantile_predictions=quantile_predictions, weights=weights)
        for policy in policies
    ]
    return pd.DataFrame([evaluation.to_dict() for evaluation in evaluations])


def evaluate_policy_by_group(
    policy: PromisePolicy,
    *,
    actual: ArrayLike,
    quantile_predictions: Mapping[float, ArrayLike],
    group_labels: Sequence[str] | pd.Series,
    weights: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute policy metrics stratified by a categorical grouping variable."""

    evaluation = evaluate_policy(
        policy,
        actual=actual,
        quantile_predictions=quantile_predictions,
        weights=weights,
    )

    actual_arr = _to_numpy(actual)
    promised_arr = _to_numpy(quantile_predictions[policy.quantile])
    groups = pd.Series(group_labels, dtype="category")
    if len(groups) != len(actual_arr):
        raise ValueError("group_labels must align with the length of the actual outcomes.")

    lateness = np.clip(actual_arr - promised_arr, a_min=0.0, a_max=None)
    earliness = np.clip(promised_arr - actual_arr, a_min=0.0, a_max=None)
    on_time = (lateness == 0).astype(float)
    cost = policy.cost_fn(actual_arr, promised_arr)

    if weights is not None:
        weight_arr = _to_numpy(weights)
        _validate_lengths(actual_arr, weight_arr)
    else:
        weight_arr = np.ones_like(actual_arr)

    df = pd.DataFrame(
        {
            "group": groups,
            "weight": weight_arr,
            "actual": actual_arr,
            "promised": promised_arr,
            "lateness": lateness,
            "earliness": earliness,
            "on_time": on_time,
            "cost": cost,
        }
    )

    grouped = df.groupby("group")
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "coverage": _weighted_mean(g["on_time"].to_numpy(), g["weight"].to_numpy()),
                "average_promised": _weighted_mean(g["promised"].to_numpy(), g["weight"].to_numpy()),
                "average_actual": _weighted_mean(g["actual"].to_numpy(), g["weight"].to_numpy()),
                "average_lateness": _weighted_mean(g["lateness"].to_numpy(), g["weight"].to_numpy()),
                "average_earliness": _weighted_mean(g["earliness"].to_numpy(), g["weight"].to_numpy()),
                "expected_cost": _weighted_mean(g["cost"].to_numpy(), g["weight"].to_numpy()),
                "sample_count": float(np.sum(g["weight"].to_numpy())),
            }
        )
    )
    summary.index.name = "group"
    summary["policy"] = evaluation.policy.display_name
    summary["quantile"] = evaluation.policy.quantile
    return summary.reset_index()


def build_fairness_report(
    group_metrics: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("coverage", "expected_cost"),
    reference_group: str | None = None,
) -> pd.DataFrame:
    """Create a disparity report comparing groups across selected metrics."""

    if "group" not in group_metrics.columns:
        raise ValueError("group_metrics must contain a 'group' column.")

    report_frames: list[pd.DataFrame] = []
    for metric in metrics:
        if metric not in group_metrics.columns:
            raise ValueError(f"Metric '{metric}' missing from group_metrics.")
        metric_values = group_metrics.set_index("group")[metric]
        if metric_values.empty:
            raise ValueError("group_metrics must contain at least one group.")
        if reference_group is not None:
            if reference_group not in metric_values.index:
                raise ValueError(f"Reference group '{reference_group}' not found.")
            reference_value = float(metric_values.loc[reference_group])
        else:
            reference_value = float(metric_values.max())
        if reference_value == 0:
            differences = metric_values.apply(lambda x: x - reference_value)
            ratios = pd.Series(np.nan, index=metric_values.index, dtype=float)
        else:
            differences = metric_values - reference_value
            ratios = metric_values / reference_value
        metric_report = pd.DataFrame(
            {
                "group": metric_values.index,
                "metric": metric,
                "value": metric_values.values,
                "difference_from_reference": differences.values,
                "ratio_to_reference": ratios.values,
            }
        )
        report_frames.append(metric_report)
    return pd.concat(report_frames, ignore_index=True)

__all__ = [
    "PromisePolicy",
    "PromiseEvaluation",
    "evaluate_policy",
    "evaluate_policies",
    "evaluate_policy_by_group",
    "build_fairness_report",
    "asymmetric_cost",
]
