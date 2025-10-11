"""Quantile regression trainers using scikit-learn models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Union

import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss

from promise_aware_eta.experiments.log_utils import log_experiment_results
from promise_aware_eta.modeling.datasets import load_experiment_splits


ConfigInput = Union[Path, str, os.PathLike[str], Mapping[str, object]]


def _resolve_config(config: ConfigInput) -> tuple[Dict, Path]:
    if isinstance(config, (str, os.PathLike)):
        config_path = Path(config)
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded: Dict = yaml.safe_load(handle)
        return loaded, config_path
    if isinstance(config, Path):
        with config.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded, config

    # Mapping input, copy to avoid accidental mutation downstream.
    loaded = dict(config)
    return loaded, Path("in-memory-config.yaml")


def train_linear_quantile(config: ConfigInput) -> Dict[float, QuantileRegressor]:
    """Train scikit-learn QuantileRegressor models for each requested quantile."""
    config_dict, config_path = _resolve_config(config)

    X_train, y_train, X_valid, y_valid, _ = load_experiment_splits(config_dict)
    quantiles = config_dict["model"]["quantiles"]
    params_cfg = config_dict["model"].get("params", {})
    if isinstance(params_cfg, dict) and "linear" in params_cfg:
        params = params_cfg["linear"] or {}
    else:
        params = params_cfg
    training_cfg = config_dict.get("training", {})

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


def train_hgb_quantile(config: ConfigInput) -> Dict[float, HistGradientBoostingRegressor]:
    """Train HistGradientBoostingRegressor in quantile mode for each quantile."""
    config_dict, config_path = _resolve_config(config)

    X_train, y_train, X_valid, y_valid, _ = load_experiment_splits(config_dict)
    quantiles = config_dict["model"]["quantiles"]
    params_cfg = config_dict["model"].get("params", {})
    if isinstance(params_cfg, dict) and "hgb" in params_cfg:
        params = params_cfg["hgb"] or {}
    else:
        params = params_cfg
    training_cfg = config_dict.get("training", {})

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


