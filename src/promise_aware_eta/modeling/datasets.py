"""Utilities for loading experiment datasets and splits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_experiment_splits(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Sequence[str]]:
    """Load training and validation splits based on an experiment config dict."""
    data_cfg = config["data"]
    features = pd.read_parquet(data_cfg["features_path"])

    feature_cols: Sequence[str] | None = data_cfg.get("feature_columns")
    if not feature_cols:
        cols_path = data_cfg.get("features_columns_path")
        if cols_path:
            feature_cols = Path(cols_path).read_text(encoding="utf-8").splitlines()
    if not feature_cols:
        exclude = {
            data_cfg["target_column"],
            "order_id",
            "order_purchase_timestamp",
        }
        feature_cols = [col for col in features.columns if col not in exclude]

    purchase_ts = pd.to_datetime(features["order_purchase_timestamp"], utc=True)
    train_period = data_cfg["train_period"]
    valid_period = data_cfg["valid_period"]
    train_mask = (purchase_ts >= pd.Timestamp(train_period["start"], tz="UTC")) & (
        purchase_ts <= pd.Timestamp(train_period["end"], tz="UTC")
    )
    valid_mask = (purchase_ts >= pd.Timestamp(valid_period["start"], tz="UTC")) & (
        purchase_ts <= pd.Timestamp(valid_period["end"], tz="UTC")
    )

    X_train = features.loc[train_mask, feature_cols]
    y_train = features.loc[train_mask, data_cfg["target_column"]]
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = features.loc[valid_mask, data_cfg["target_column"]]
    return X_train, y_train, X_valid, y_valid, feature_cols
