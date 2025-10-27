"""Tests for policy simulation and fairness utilities."""

from __future__ import annotations

import numpy as np
import pytest

from promise_aware_eta.policy import (
    PromisePolicy,
    asymmetric_cost,
    build_fairness_report,
    evaluate_policies,
    evaluate_policy,
    evaluate_policy_by_group,
)


def _simple_cost(actual: np.ndarray, promised: np.ndarray) -> np.ndarray:
    diff = actual - promised
    late = np.where(diff > 0, diff * 2.0, 0.0)
    early = np.where(diff < 0, -diff, 0.0)
    return late + early


def test_evaluate_policy_basic_metrics() -> None:
    actual = np.array([2.0, 3.0, 5.0])
    predictions = {0.8: np.array([3.0, 3.0, 4.0])}
    policy = PromisePolicy(quantile=0.8, cost_fn=_simple_cost, name="p80")

    evaluation = evaluate_policy(policy, actual=actual, quantile_predictions=predictions)

    assert evaluation.coverage == pytest.approx(2 / 3)
    assert evaluation.average_promised == pytest.approx(10 / 3)
    assert evaluation.average_actual == pytest.approx(10 / 3)
    assert evaluation.average_lateness == pytest.approx(1 / 3)
    assert evaluation.average_earliness == pytest.approx(1 / 3)
    assert evaluation.expected_cost == pytest.approx(1.0)


def test_evaluate_policies_returns_dataframe() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = {
        0.5: np.array([2.0, 2.0, 3.0, 3.0]),
        0.9: np.array([3.0, 3.0, 4.0, 5.0]),
    }
    policies = [
        PromisePolicy(quantile=0.5, cost_fn=asymmetric_cost, name="p50"),
        PromisePolicy(quantile=0.9, cost_fn=asymmetric_cost, name="p90"),
    ]

    results = evaluate_policies(policies, actual=actual, quantile_predictions=predictions)

    assert set(results.columns) == {
        "policy",
        "quantile",
        "coverage",
        "average_promised",
        "average_actual",
        "average_lateness",
        "average_earliness",
        "expected_cost",
    }
    assert len(results) == 2
    assert results.loc[results["policy"] == "p50", "quantile"].iloc[0] == 0.5


def test_group_fairness_metrics_and_report() -> None:
    actual = np.array([2.0, 3.0, 4.5, 5.0])
    predictions = {0.8: np.array([2.5, 3.5, 4.0, 5.0])}
    policy = PromisePolicy(quantile=0.8, cost_fn=_simple_cost, name="p80")
    groups = ["north", "south", "north", "south"]

    group_metrics = evaluate_policy_by_group(
        policy,
        actual=actual,
        quantile_predictions=predictions,
        group_labels=groups,
    )

    assert set(group_metrics.columns) >= {
        "group",
        "coverage",
        "average_lateness",
        "expected_cost",
        "policy",
        "quantile",
    }
    assert {"north", "south"} == set(group_metrics["group"])

    report = build_fairness_report(group_metrics, metrics=("coverage", "expected_cost"), reference_group="north")
    assert set(report["metric"]) == {"coverage", "expected_cost"}
    north_rows = report[report["group"] == "north"]
    assert (north_rows["difference_from_reference"].abs() < 1e-9).all()
