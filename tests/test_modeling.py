import numpy as np
import pandas as pd

from pathlib import Path

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
