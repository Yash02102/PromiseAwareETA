"""Quantile regression trainers using scikit-learn models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss

from promise_aware_eta.experiments.log_utils import log_experiment_results
from promise_aware_eta.modeling.datasets import load_experiment_splits


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
    for quantile in quantiles:
        model = QuantileRegressor(quantile=quantile, **params)
        model.fit(X_train, y_train)
        loss = float("nan")
        if training_cfg.get("report_metrics", True):
            valid_pred = model.predict(X_valid)
            loss = float(mean_pinball_loss(y_valid, valid_pred, alpha=quantile))
            print(f"Quantile {quantile}: validation pinball loss {loss:.4f}")
        metrics.append({"quantile": float(quantile), "pinball_loss": loss})
        models[quantile] = model

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
    for quantile in quantiles:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **params)
        model.fit(X_train, y_train)
        loss = float("nan")
        if training_cfg.get("report_metrics", True):
            valid_pred = model.predict(X_valid)
            loss = float(mean_pinball_loss(y_valid, valid_pred, alpha=quantile))
            print(f"Quantile {quantile}: validation pinball loss {loss:.4f}")
        metrics.append({"quantile": float(quantile), "pinball_loss": loss})
        models[quantile] = model

    log_experiment_results(
        model="hgb",
        config_path=config_path,
        metrics=metrics,
        train_rows=len(X_train),
        valid_rows=len(X_valid),
        feature_columns=X_train.columns,
    )
    return models


