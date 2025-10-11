from pathlib import Path

import numpy as np
import pandas as pd

from promise_aware_eta.modeling import QuantileModelSpec, train_quantile_model
from promise_aware_eta.modeling.datasets import load_experiment_splits
from promise_aware_eta.modeling.sklearn_quantile import (
    train_hgb_quantile,
    train_linear_quantile,
)


def _make_synthetic_features(tmp_path: Path) -> Path:
    path = tmp_path / "features.parquet"
    n = 60
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "order_id": [f"order_{i}" for i in range(n)],
            "order_purchase_timestamp": pd.date_range("2017-01-01", periods=n, freq="D"),
            "delivery_lag_days": rng.normal(loc=10, scale=2, size=n),
            "feat_price": rng.uniform(50, 200, size=n),
            "feat_distance": rng.uniform(0, 1, size=n),
        }
    )
    df.to_parquet(path, index=False)
    cols_path = tmp_path / "features_columns.txt"
    cols_path.write_text("feat_price\nfeat_distance", encoding="utf-8")
    return path


def _base_config(tmp_path: Path) -> Path:
    features_path = _make_synthetic_features(tmp_path)
    config = {
        "model": {
            "quantiles": [0.5],
            "params": {},
        },
        "data": {
            "features_path": str(features_path),
            "target_column": "delivery_lag_days",
            "features_columns_path": str(tmp_path / "features_columns.txt"),
            "train_period": {"start": "2017-01-01", "end": "2017-02-15"},
            "valid_period": {"start": "2017-02-16", "end": "2017-03-02"},
        },
        "training": {"report_metrics": False},
    }
    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_experiment_splits_returns_expected_shapes(tmp_path):
    import yaml

    config_path = _base_config(tmp_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    X_train, y_train, X_valid, y_valid, feature_cols = load_experiment_splits(cfg)
    assert len(feature_cols) == 2
    assert not X_train.empty and not X_valid.empty
    assert set(X_train.columns) == set(feature_cols)
    assert y_train.name == "delivery_lag_days"


def test_train_linear_quantile_runs(tmp_path):
    config_path = _base_config(tmp_path)
    models = train_linear_quantile(config_path)
    assert 0.5 in models


def test_train_hgb_quantile_runs(tmp_path):
    config_path = _base_config(tmp_path)
    models = train_hgb_quantile(config_path)
    assert 0.5 in models


def _dispatcher_dataset() -> dict:
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    feature_columns = ["feat_price", "feat_distance"]

    df = pd.DataFrame(
        {
            "order_id": [f"order_{i}" for i in range(len(dates))],
            "order_purchase_timestamp": dates,
            "feat_price": rng.uniform(50, 200, size=len(dates)),
            "feat_distance": rng.uniform(0, 1, size=len(dates)),
            "delivery_lag_days": rng.normal(loc=10, scale=1.5, size=len(dates)),
        }
    )

    return {
        "features_df": df,
        "target_column": "delivery_lag_days",
        "feature_columns": feature_columns,
        "train_period": {"start": "2017-01-01", "end": "2017-02-20"},
        "valid_period": {"start": "2017-02-21", "end": "2017-03-02"},
    }


def _build_spec(model_name: str, params: dict | None = None) -> QuantileModelSpec:
    return QuantileModelSpec(
        name=model_name,
        quantiles=[0.1, 0.5, 0.9],
        params=params or {},
        data=_dispatcher_dataset(),
        training={"report_metrics": False, "num_boost_round": 5},
    )


def test_quantile_dispatcher_linear():
    models = train_quantile_model(_build_spec("linear"))
    assert set(models.keys()) == {0.1, 0.5, 0.9}


def test_quantile_dispatcher_hgb():
    params = {"hgb": {"max_iter": 5}}
    models = train_quantile_model(_build_spec("hgb", params=params))
    assert set(models.keys()) == {0.1, 0.5, 0.9}


def test_quantile_dispatcher_lightgbm():
    params = {
        "lightgbm": {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "verbose": -1,
        }
    }
    spec = _build_spec("lightgbm", params=params)
    models = train_quantile_model(spec)
    assert set(models.keys()) == {0.1, 0.5, 0.9}
