"""Run staged experiments: baselines, calibration, policy simulation, and fairness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
import yaml

from promise_aware_eta.modeling import QuantileModelSpec, train_quantile_model
from promise_aware_eta.modeling.calibration import (
    SplitConformalQuantileCalibrator,
    compute_coverage_diagnostics,
)
from promise_aware_eta.modeling.datasets import load_experiment_splits
from promise_aware_eta.policy import (
    PromisePolicy,
    asymmetric_cost,
    build_fairness_report,
    evaluate_policies,
    evaluate_policy_by_group,
)

try:  # pragma: no cover - import resolution flexibility for scripts/tests.
    from .run_synthetic_suite import (
        CONFIG_PATH,
        FEATURES_PATH,
        generate_synthetic_dataset,
    )
except ImportError:  # pragma: no cover - executed when module loaded without package context.
    from experiments.run_synthetic_suite import (  # type: ignore[no-redef]
        CONFIG_PATH,
        FEATURES_PATH,
        generate_synthetic_dataset,
    )

RESULTS_DIR = Path("experiments/logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FAIRNESS_MARKDOWN_PATH = RESULTS_DIR / "stage4_region_fairness.md"


def _fairness_report_markdown(report: pd.DataFrame) -> str:
    """Return a compact markdown table summarizing disparity metrics."""

    if report.empty:
        raise ValueError("Fairness report must contain at least one row.")

    pivot = report.pivot(
        index="group",
        columns="metric",
        values=["value", "difference_from_reference", "ratio_to_reference"],
    )
    required_metrics = {"coverage", "expected_cost"}
    available_metrics = set(pivot.columns.get_level_values(1))
    if not required_metrics.issubset(available_metrics):
        missing = required_metrics.difference(available_metrics)
        raise ValueError(f"Fairness report missing required metrics: {sorted(missing)}")

    coverage = pivot[("value", "coverage")]
    coverage_delta = pivot[("difference_from_reference", "coverage")]
    cost_value = pivot[("value", "expected_cost")]
    cost_ratio = pivot[("ratio_to_reference", "expected_cost")]

    summary = pd.DataFrame(
        {
            "Group": coverage.index,
            "Coverage": coverage.to_numpy(),
            "Δ Coverage": coverage_delta.to_numpy(),
            "Expected Cost": cost_value.to_numpy(),
            "Cost Ratio": cost_ratio.to_numpy(),
        }
    )

    summary = summary.sort_values("Coverage", ascending=False)

    def _fmt(value: float) -> str:
        return f"{value:.3f}"

    header = ["Group", "Coverage", "Δ Coverage", "Expected Cost", "Cost Ratio"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Group"]),
                    _fmt(row["Coverage"]),
                    _fmt(row["Δ Coverage"]),
                    _fmt(row["Expected Cost"]),
                    _fmt(row["Cost Ratio"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_validation_metadata(config: dict) -> pd.DataFrame:
    full_df = pd.read_parquet(FEATURES_PATH)
    purchase_ts = pd.to_datetime(full_df["order_purchase_timestamp"], utc=True)
    valid_period = config["data"]["valid_period"]
    valid_mask = (purchase_ts >= pd.Timestamp(valid_period["start"], tz="UTC")) & (
        purchase_ts <= pd.Timestamp(valid_period["end"], tz="UTC")
    )
    return full_df.loc[valid_mask].reset_index(drop=True)


def run_stage1_baselines(config: dict) -> Dict[float, object]:
    """Train the linear baseline quantile models for reuse in later stages."""

    spec = QuantileModelSpec(
        name="linear",
        quantiles=config["model"]["quantiles"],
        params=config["model"].get("params", {}),
        data=config["data"],
        training=config.get("training", {}),
    )
    print("=== Stage 1: training linear quantile baseline ===")
    models = train_quantile_model(spec)
    return models


def run_stage2_calibration(models: Dict[float, object], config: dict) -> dict[str, float]:
    """Perform split-conformal calibration on validation predictions."""

    _, _, X_valid, y_valid, _ = load_experiment_splits(config)
    if len(y_valid) < 10:
        raise RuntimeError("Validation split too small for calibration.")

    lower_q, upper_q = min(models.keys()), max(models.keys())
    lower_model = models[lower_q]
    upper_model = models[upper_q]

    lower_pred = np.asarray(lower_model.predict(X_valid), dtype=float)
    upper_pred = np.asarray(upper_model.predict(X_valid), dtype=float)

    calib_size = len(y_valid) // 2
    y_calib = y_valid.iloc[:calib_size]
    y_eval = y_valid.iloc[calib_size:]
    lower_calib = lower_pred[:calib_size]
    upper_calib = upper_pred[:calib_size]
    lower_eval = lower_pred[calib_size:]
    upper_eval = upper_pred[calib_size:]

    target_coverage = config["model"].get("target_coverage", 0.9)
    alpha = 1 - target_coverage

    calibrator = SplitConformalQuantileCalibrator(alpha=alpha)
    print("=== Stage 2: split-conformal calibration ===")
    adjusted = calibrator.calibrate(
        y_calib=y_calib,
        lower_calib=lower_calib,
        upper_calib=upper_calib,
        lower_pred=lower_eval,
        upper_pred=upper_eval,
    )
    diagnostics = compute_coverage_diagnostics(
        y_true=y_eval,
        lower=adjusted["lower"],
        upper=adjusted["upper"],
        target_coverage=target_coverage,
    )
    payload = diagnostics.as_dict()
    (RESULTS_DIR / "stage2_calibration_diagnostics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def run_stage3_policy_simulation(
    models: Dict[float, object],
    config: dict,
    *,
    policy_quantiles: Tuple[float, ...] = (0.5, 0.8, 0.9),
) -> tuple[pd.DataFrame, Dict[float, np.ndarray], np.ndarray]:
    """Apply the offline policy simulator using validation predictions."""

    _, _, X_valid, y_valid, _ = load_experiment_splits(config)
    predictions = {
        quantile: np.asarray(model.predict(X_valid), dtype=float)
        for quantile, model in models.items()
        if hasattr(model, "predict")
    }
    actual = y_valid.to_numpy(dtype=float)
    policies = [
        PromisePolicy(quantile=q, cost_fn=asymmetric_cost, name=f"p{int(q * 100)}")
        for q in policy_quantiles
        if q in predictions
    ]
    if not policies:
        raise RuntimeError("No matching quantile predictions for requested policies.")

    print("=== Stage 3: offline policy simulation ===")
    results = evaluate_policies(policies, actual=actual, quantile_predictions=predictions)
    results_path = RESULTS_DIR / "stage3_policy_metrics.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved policy metrics to {results_path}")
    return results, predictions, actual


def run_stage4_fairness(
    policy: PromisePolicy,
    predictions: Dict[float, np.ndarray],
    actual: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seller and region fairness reports for the selected policy."""

    if len(metadata) != len(actual):
        raise ValueError("Metadata and actual arrays must align in length.")

    seller_metrics = evaluate_policy_by_group(
        policy,
        actual=actual,
        quantile_predictions=predictions,
        group_labels=metadata["seller_id"],
    )

    region_metrics = evaluate_policy_by_group(
        policy,
        actual=actual,
        quantile_predictions=predictions,
        group_labels=metadata["seller_region"],
    )

    region_report = build_fairness_report(region_metrics, reference_group=None)

    seller_metrics.to_csv(RESULTS_DIR / "stage4_seller_metrics.csv", index=False)
    region_report.to_csv(RESULTS_DIR / "stage4_region_fairness.csv", index=False)
    markdown = _fairness_report_markdown(region_report)
    FAIRNESS_MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    print("Saved fairness reports to experiments/logs/.")
    return seller_metrics, region_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-regen", action="store_true", help="Regenerate the synthetic dataset")
    return parser.parse_args()


def main(force_regen: bool = False) -> None:
    generate_synthetic_dataset(force=force_regen)
    config = _load_config()
    metadata = _load_validation_metadata(config)
    models = run_stage1_baselines(config)
    diagnostics = run_stage2_calibration(models, config)
    print("Calibration diagnostics:", diagnostics)
    policy_metrics, predictions, actual = run_stage3_policy_simulation(models, config)

    # Choose the policy whose achieved coverage is closest to the target for fairness review.
    target = config["model"].get("target_coverage", 0.9)
    coverage_values = pd.to_numeric(policy_metrics["coverage"], errors="coerce")
    valid_coverage = coverage_values.dropna()
    if valid_coverage.empty:
        raise RuntimeError("Policy metrics did not include valid coverage values.")

    selected_idx = (valid_coverage - target).abs().idxmin()
    selected_row = policy_metrics.loc[selected_idx]
    target_quantile = float(selected_row["quantile"])
    policy = PromisePolicy(quantile=target_quantile, cost_fn=asymmetric_cost)
    if policy.quantile not in predictions:
        raise RuntimeError(f"No predictions available for quantile {policy.quantile}.")
    run_stage4_fairness(policy, predictions, actual, metadata)


if __name__ == "__main__":
    args = parse_args()
    main(force_regen=args.force_regen)
