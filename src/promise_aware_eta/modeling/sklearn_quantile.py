"""Quantile regression trainers using scikit-learn models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss

from promise_aware_eta.experiments.log_utils import log_experiment_results
from promise_aware_eta.modeling.datasets import load_experiment_splits
from promise_aware_eta.modeling.conformal import (
    RollingConformalDiagnostics,
    rolling_split_conformal,
)


def train_linear_quantile(config_path: Path) -> Dict[float, QuantileRegressor]:
    """Train scikit-learn QuantileRegressor models for each requested quantile."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config: Dict = yaml.safe_load(handle)

    X_train, y_train, X_valid, y_valid, _ = load_experiment_splits(config)
    quantiles = config["model"]["quantiles"]
    params_cfg = config["model"].get("params", {})
    if isinstance(params_cfg, dict) and "linear" in params_cfg:
        params = params_cfg["linear"] or {}
    else:
        params = params_cfg
    training_cfg = config.get("training", {})

    models: Dict[float, QuantileRegressor] = {}
    metrics: List[dict] = []
    quantile_losses: Dict[float, float] = {}
    valid_predictions: Dict[float, np.ndarray] = {}
    for quantile in quantiles:
        model = QuantileRegressor(quantile=quantile, **params)
        model.fit(X_train, y_train)
        loss = float("nan")
        valid_pred = model.predict(X_valid)
        valid_predictions[quantile] = valid_pred
        if training_cfg.get("report_metrics", True):
            loss = float(mean_pinball_loss(y_valid, valid_pred, alpha=quantile))
            print(f"Quantile {quantile}: validation pinball loss {loss:.4f}")
        quantile_losses[quantile] = loss
        models[quantile] = model

    calibration_cfg = training_cfg.get("calibration") or {}
    calibrated_pairs: set[Tuple[float, float]] = set()
    calibration_diagnostics: List[RollingConformalDiagnostics] = []
    if calibration_cfg.get("method") == "split_conformal":
        interval_pairs = calibration_cfg.get("intervals") or []
        if not interval_pairs and len(valid_predictions) >= 2:
            sorted_q = sorted(valid_predictions)
            interval_pairs = [[float(sorted_q[0]), float(sorted_q[-1])]]
        target_coverage = float(calibration_cfg.get("target_coverage", 0.9))
        num_splits = int(calibration_cfg.get("rolling_splits", 0))
        if interval_pairs:
            y_valid_values = np.asarray(y_valid)
            for pair in interval_pairs:
                lower_q, upper_q = sorted(float(q) for q in pair)
                if lower_q not in valid_predictions or upper_q not in valid_predictions:
                    continue
                lower_preds = valid_predictions[lower_q]
                upper_preds = valid_predictions[upper_q]
                lower_adj, upper_adj, diag = rolling_split_conformal(
                    y_true=y_valid_values,
                    lower=lower_preds,
                    upper=upper_preds,
                    target_coverage=target_coverage,
                    lower_quantile=lower_q,
                    upper_quantile=upper_q,
                    num_splits=num_splits,
                )
                valid_predictions[lower_q] = lower_adj
                valid_predictions[upper_q] = upper_adj
                calibrated_pairs.add((lower_q, upper_q))
                calibration_diagnostics.append(diag)

    y_valid_values = np.asarray(y_valid)
    sorted_quantiles: Sequence[float] = sorted(valid_predictions)
    for quantile in sorted_quantiles:
        preds = valid_predictions[quantile]
        coverage = float(np.mean(y_valid_values <= preds))
        coverage_error = coverage - float(quantile)
        metrics.append(
            {
                "type": "quantile",
                "quantile": float(quantile),
                "pinball_loss": float(quantile_losses.get(quantile, float("nan"))),
                "coverage": coverage,
                "coverage_error": coverage_error,
            }
        )

    for idx, lower_q in enumerate(sorted_quantiles):
        for upper_q in sorted_quantiles[idx + 1 :]:
            lower_preds = valid_predictions[lower_q]
            upper_preds = valid_predictions[upper_q]
            observed_coverage = float(
                np.mean(
                    (y_valid_values >= lower_preds) & (y_valid_values <= upper_preds)
                )
            )
            mean_width = float(np.mean(upper_preds - lower_preds))
            target = float(upper_q - lower_q)
            metrics.append(
                {
                    "type": "interval",
                    "lower_quantile": float(lower_q),
                    "upper_quantile": float(upper_q),
                    "target_coverage": target,
                    "observed_coverage": observed_coverage,
                    "coverage_error": observed_coverage - target,
                    "mean_width": mean_width,
                    "calibrated": (lower_q, upper_q) in calibrated_pairs,
                }
            )

    for diag in calibration_diagnostics:
        metrics.append(
            {
                "type": "calibration_diagnostics",
                "lower_quantile": diag.lower_quantile,
                "upper_quantile": diag.upper_quantile,
                "target_coverage": diag.target_coverage,
                "num_splits": diag.num_splits,
                "adjustments": diag.adjustments,
                "split_coverages": diag.split_coverages,
                "split_widths": diag.split_widths,
            }
        )

    log_experiment_results(
        model="linear",
        config_path=config_path,
        metrics=metrics,
        train_rows=len(X_train),
        valid_rows=len(X_valid),
        feature_columns=X_train.columns,
    )
    return models


def train_hgb_quantile(config_path: Path) -> Dict[float, HistGradientBoostingRegressor]:
    """Train HistGradientBoostingRegressor in quantile mode for each quantile."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config: Dict = yaml.safe_load(handle)

    X_train, y_train, X_valid, y_valid, _ = load_experiment_splits(config)
    quantiles = config["model"]["quantiles"]
    params_cfg = config["model"].get("params", {})
    if isinstance(params_cfg, dict) and "hgb" in params_cfg:
        params = params_cfg["hgb"] or {}
    else:
        params = params_cfg
    training_cfg = config.get("training", {})

    models: Dict[float, HistGradientBoostingRegressor] = {}
    metrics: List[dict] = []
    quantile_losses: Dict[float, float] = {}
    valid_predictions: Dict[float, np.ndarray] = {}
    for quantile in quantiles:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **params)
        model.fit(X_train, y_train)
        loss = float("nan")
        valid_pred = model.predict(X_valid)
        valid_predictions[quantile] = valid_pred
        if training_cfg.get("report_metrics", True):
            loss = float(mean_pinball_loss(y_valid, valid_pred, alpha=quantile))
            print(f"Quantile {quantile}: validation pinball loss {loss:.4f}")
        quantile_losses[quantile] = loss
        models[quantile] = model

    calibration_cfg = training_cfg.get("calibration") or {}
    calibrated_pairs: set[Tuple[float, float]] = set()
    calibration_diagnostics: List[RollingConformalDiagnostics] = []
    if calibration_cfg.get("method") == "split_conformal":
        interval_pairs = calibration_cfg.get("intervals") or []
        if not interval_pairs and len(valid_predictions) >= 2:
            sorted_q = sorted(valid_predictions)
            interval_pairs = [[float(sorted_q[0]), float(sorted_q[-1])]]
        target_coverage = float(calibration_cfg.get("target_coverage", 0.9))
        num_splits = int(calibration_cfg.get("rolling_splits", 0))
        if interval_pairs:
            y_valid_values = np.asarray(y_valid)
            for pair in interval_pairs:
                lower_q, upper_q = sorted(float(q) for q in pair)
                if lower_q not in valid_predictions or upper_q not in valid_predictions:
                    continue
                lower_preds = valid_predictions[lower_q]
                upper_preds = valid_predictions[upper_q]
                lower_adj, upper_adj, diag = rolling_split_conformal(
                    y_true=y_valid_values,
                    lower=lower_preds,
                    upper=upper_preds,
                    target_coverage=target_coverage,
                    lower_quantile=lower_q,
                    upper_quantile=upper_q,
                    num_splits=num_splits,
                )
                valid_predictions[lower_q] = lower_adj
                valid_predictions[upper_q] = upper_adj
                calibrated_pairs.add((lower_q, upper_q))
                calibration_diagnostics.append(diag)

    y_valid_values = np.asarray(y_valid)
    sorted_quantiles: Sequence[float] = sorted(valid_predictions)
    for quantile in sorted_quantiles:
        preds = valid_predictions[quantile]
        coverage = float(np.mean(y_valid_values <= preds))
        coverage_error = coverage - float(quantile)
        metrics.append(
            {
                "type": "quantile",
                "quantile": float(quantile),
                "pinball_loss": float(quantile_losses.get(quantile, float("nan"))),
                "coverage": coverage,
                "coverage_error": coverage_error,
            }
        )

    for idx, lower_q in enumerate(sorted_quantiles):
        for upper_q in sorted_quantiles[idx + 1 :]:
            lower_preds = valid_predictions[lower_q]
            upper_preds = valid_predictions[upper_q]
            observed_coverage = float(
                np.mean(
                    (y_valid_values >= lower_preds) & (y_valid_values <= upper_preds)
                )
            )
            mean_width = float(np.mean(upper_preds - lower_preds))
            target = float(upper_q - lower_q)
            metrics.append(
                {
                    "type": "interval",
                    "lower_quantile": float(lower_q),
                    "upper_quantile": float(upper_q),
                    "target_coverage": target,
                    "observed_coverage": observed_coverage,
                    "coverage_error": observed_coverage - target,
                    "mean_width": mean_width,
                    "calibrated": (lower_q, upper_q) in calibrated_pairs,
                }
            )

    for diag in calibration_diagnostics:
        metrics.append(
            {
                "type": "calibration_diagnostics",
                "lower_quantile": diag.lower_quantile,
                "upper_quantile": diag.upper_quantile,
                "target_coverage": diag.target_coverage,
                "num_splits": diag.num_splits,
                "adjustments": diag.adjustments,
                "split_coverages": diag.split_coverages,
                "split_widths": diag.split_widths,
            }
        )

    log_experiment_results(
        model="hgb",
        config_path=config_path,
        metrics=metrics,
        train_rows=len(X_train),
        valid_rows=len(X_valid),
        feature_columns=X_train.columns,
    )
    return models


