from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_pinball_loss

from promise_aware_eta.modeling import QuantileModelSpec, train_quantile_model
from promise_aware_eta.modeling.calibration import (
    SplitConformalQuantileCalibrator,
    compute_coverage_diagnostics,
)
from promise_aware_eta.modeling.datasets import load_experiment_splits
from promise_aware_eta.policy import (
    PromisePolicy,
    asymmetric_cost,
    evaluate_policies,
    evaluate_policy_by_group,
)

from experiments.run_synthetic_suite import CONFIG_PATH, generate_synthetic_dataset

OUTPUT_DIR = Path("analysis/figures/data")
TARGET_COVERAGES: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
MODEL_NAMES: tuple[str, ...] = ("linear", "hgb", "lightgbm")


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_validation_frame(config: dict) -> pd.DataFrame:
    data_cfg = config["data"].copy()
    features_path = Path(data_cfg.get("features_path"))
    full_df = pd.read_parquet(features_path)
    purchase_ts = pd.to_datetime(full_df["order_purchase_timestamp"], utc=True)
    valid_period = data_cfg["valid_period"]
    valid_mask = (purchase_ts >= pd.Timestamp(valid_period["start"], tz="UTC")) & (
        purchase_ts <= pd.Timestamp(valid_period["end"], tz="UTC")
    )
    return full_df.loc[valid_mask].reset_index(drop=True)


def _compute_pinball_metrics(
    model_name: str,
    estimators: Dict[float, object],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, Dict[float, np.ndarray]]:
    records: list[dict] = []
    predictions: Dict[float, np.ndarray] = {}
    for quantile, estimator in estimators.items():
        pred = estimator.predict(X_valid)
        predictions[quantile] = np.asarray(pred, dtype=float)
        loss = float(mean_pinball_loss(y_valid, pred, alpha=quantile))
        records.append(
            {
                "model": model_name,
                "quantile": float(quantile),
                "pinball_loss": loss,
            }
        )
    return pd.DataFrame.from_records(records), predictions


def _compute_interval_summary(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    y_valid: pd.Series,
) -> pd.DataFrame:
    calib_size = len(y_valid) // 2
    y_calib = y_valid.iloc[:calib_size]
    y_eval = y_valid.iloc[calib_size:]
    lower_calib = lower_pred[:calib_size]
    upper_calib = upper_pred[:calib_size]
    lower_eval = lower_pred[calib_size:]
    upper_eval = upper_pred[calib_size:]

    rows: list[dict] = []
    for target in TARGET_COVERAGES:
        alpha = 1.0 - target
        calibrator = SplitConformalQuantileCalibrator(alpha=alpha)
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
            target_coverage=target,
        )
        rows.append(
            {
                "target_coverage": target,
                "empirical_coverage": diagnostics.coverage,
                "mean_interval_width": diagnostics.mean_interval_width,
                "conformal_quantile": calibrator.conformal_quantile_,
            }
        )
    return pd.DataFrame(rows)


def _prepare_segment_keys(valid_df: pd.DataFrame) -> pd.Series:
    distance_rank = valid_df["distance_km"].rank(pct=True, method="average")
    dispatch_rank = valid_df["dispatch_delay_days"].rank(pct=True, method="average")
    distance_decile = np.clip(np.ceil(distance_rank * 10).astype(int), 1, 10)
    dispatch_decile = np.clip(np.ceil(dispatch_rank * 10).astype(int), 1, 10)
    return pd.Series(
        [f"d{d:02d}_s{s:02d}" for d, s in zip(distance_decile, dispatch_decile)],
        index=valid_df.index,
        name="segment_key",
    )


def _compute_feature_importances(
    lightgbm_models: Dict[float, object]
) -> pd.DataFrame:
    rows: list[dict] = []
    for quantile, booster in lightgbm_models.items():
        feature_names: Iterable[str] = booster.feature_name()
        importances = booster.feature_importance(importance_type="gain")
        total = float(importances.sum())
        for name, importance in zip(feature_names, importances):
            gain = float(importance)
            pct = gain / total if total > 0 else 0.0
            rows.append(
                {
                    "quantile": float(quantile),
                    "feature": name,
                    "importance_gain": gain,
                    "importance_gain_pct": pct,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_synthetic_dataset(force=False)
    config = _load_config()

    _, _, X_valid, y_valid, _ = load_experiment_splits(config)
    valid_df = _load_validation_frame(config)

    all_metrics: list[pd.DataFrame] = []
    model_predictions: Dict[str, Dict[float, np.ndarray]] = {}
    fitted_models: Dict[str, Dict[float, object]] = {}

    for model_name in MODEL_NAMES:
        spec = QuantileModelSpec(
            name=model_name,
            quantiles=config["model"]["quantiles"],
            params=config["model"].get("params", {}),
            data=config["data"],
            training=config.get("training", {}),
        )
        estimators = train_quantile_model(spec)
        metrics_df, predictions = _compute_pinball_metrics(
            model_name,
            estimators,
            X_valid,
            y_valid,
        )
        all_metrics.append(metrics_df)
        model_predictions[model_name] = predictions
        fitted_models[model_name] = estimators

    pinball_df = pd.concat(all_metrics, ignore_index=True).sort_values(
        ["model", "quantile"]
    )
    pinball_df.to_csv(OUTPUT_DIR / "F1_pinball_loss.csv", index=False)

    linear_preds = model_predictions["linear"]
    lower_q = min(linear_preds)
    upper_q = max(linear_preds)
    interval_df = _compute_interval_summary(
        lower_pred=linear_preds[lower_q],
        upper_pred=linear_preds[upper_q],
        y_valid=y_valid,
    )
    interval_df.to_csv(OUTPUT_DIR / "F3_interval_width_vs_coverage.csv", index=False)

    policy_quantiles = sorted(linear_preds.keys())
    policies = [
        PromisePolicy(quantile=q, cost_fn=asymmetric_cost, name=f"p{int(q * 100)}")
        for q in policy_quantiles
    ]
    policy_metrics = evaluate_policies(
        policies,
        actual=y_valid,
        quantile_predictions=linear_preds,
    )
    policy_metrics = policy_metrics.assign(
        late_rate=lambda df: 1.0 - df["coverage"],
        avg_promise_days=lambda df: df["average_promised"],
        tau=lambda df: df["quantile"],
    )[
        ["tau", "late_rate", "avg_promise_days", "coverage", "expected_cost", "policy"]
    ]
    policy_metrics.to_csv(OUTPUT_DIR / "F5_lpr_apd_frontier.csv", index=False)

    segment_keys = _prepare_segment_keys(valid_df)
    policy = PromisePolicy(quantile=upper_q, cost_fn=asymmetric_cost, name=f"p{int(upper_q * 100)}")
    segment_metrics = evaluate_policy_by_group(
        policy,
        actual=y_valid,
        quantile_predictions=linear_preds,
        group_labels=segment_keys,
    )
    segment_metrics = segment_metrics.rename(columns={"group": "segment_key"}).assign(
        tau=policy.quantile,
        late_rate=lambda df: 1.0 - df["coverage"],
    )[
        [
            "segment_key",
            "tau",
            "coverage",
            "late_rate",
            "average_promised",
            "average_actual",
            "expected_cost",
            "sample_count",
        ]
    ]
    segment_metrics.to_csv(OUTPUT_DIR / "F6_segment_heatmap.csv", index=False)

    feature_importances = _compute_feature_importances(fitted_models["lightgbm"])
    feature_importances.to_csv(OUTPUT_DIR / "F9_feature_contributions.csv", index=False)

    promised = linear_preds[upper_q]
    actual = y_valid.to_numpy(dtype=float)
    lateness = np.clip(actual - promised, a_min=0.0, a_max=None)
    earliness = np.clip(promised - actual, a_min=0.0, a_max=None)
    late_flag = (lateness > 0).astype(int)

    violation_df = pd.DataFrame(
        {
            "order_id": valid_df["order_id"],
            "corridor_km": valid_df["distance_km"],
            "dispatch_delay_days": valid_df["dispatch_delay_days"],
            "tau": upper_q,
            "promised_days": promised,
            "actual_days": actual,
            "lateness_days": lateness,
            "earliness_days": earliness,
            "late_flag": late_flag,
        }
    )
    violation_df.to_csv(OUTPUT_DIR / "F10_violation_vs_distance.csv", index=False)

    purchase_ts = pd.to_datetime(valid_df["order_purchase_timestamp"], utc=True)
    coverage_series = 1.0 - late_flag
    week_start = purchase_ts.dt.to_period("W-MON").dt.start_time
    temporal_df = (
        pd.DataFrame(
            {
                "week_start": week_start,
                "on_time": coverage_series,
            }
        )
        .groupby("week_start")
        .agg(empirical_coverage=("on_time", "mean"), sample_count=("on_time", "size"))
        .reset_index()
        .sort_values("week_start")
    )
    temporal_df.to_csv(OUTPUT_DIR / "F11_temporal_drift.csv", index=False)

    avg_pinball = (
        pinball_df.groupby("model")["pinball_loss"].mean().rename("mean_pinball_loss")
    )
    baseline = float(avg_pinball.loc["linear"])
    ablation_df = (
        avg_pinball.reset_index()
        .assign(delta_pinball=lambda df: df["mean_pinball_loss"] - baseline)
        .rename(columns={"model": "family"})
    )
    ablation_df.to_csv(OUTPUT_DIR / "F12_ablation_tornado.csv", index=False)


if __name__ == "__main__":
    main()
