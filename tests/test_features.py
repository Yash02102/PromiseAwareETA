from pathlib import Path

import pandas as pd

from promise_aware_eta.data_ingestion import load_olist_tables
from promise_aware_eta.features import build_model_features
from promise_aware_eta.pipelines.build_features import main as build_features_main, FEATURES_PARQUET, FEATURE_COLUMNS_TXT


def test_build_model_features_outputs_expected_columns(sample_raw_dir):
    tables = load_olist_tables(sample_raw_dir)
    features = build_model_features(tables)
    assert "order_id" in features.columns
    expected_columns = {
        "basket_items",
        "basket_sellers",
        "basket_products",
        "basket_price_sum",
        "basket_freight_sum",
        "payment_installments_max",
        "payment_types",
        "payment_value_sum",
        "purchase_hour",
        "purchase_dow",
        "purchase_week",
        "purchase_month",
    }
    assert expected_columns.issubset(set(features.columns))


def test_build_features_pipeline_creates_artifacts(tmp_path, monkeypatch, sample_raw_dir):
    output_dir = tmp_path / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PROMISE_FEATURE_RAW_DIR", str(sample_raw_dir))

    # Monkeypatch module-level constants
    from importlib import reload
    import promise_aware_eta.pipelines.build_features as build_features

    build_features.RAW_DIR = sample_raw_dir
    build_features.OUTPUT_DIR = output_dir
    build_features.FEATURES_PARQUET = output_dir / "features.parquet"
    build_features.FEATURE_COLUMNS_TXT = output_dir / "features_columns.txt"

    build_features_main(sample_raw_dir)

    assert build_features.FEATURES_PARQUET.exists()
    assert build_features.FEATURE_COLUMNS_TXT.exists()

    df = pd.read_parquet(build_features.FEATURES_PARQUET)
    feature_columns = build_features.FEATURE_COLUMNS_TXT.read_text().splitlines()
    for col in feature_columns:
        assert col in df.columns

    assert "delivery_lag_days" in df.columns
    assert "order_purchase_timestamp" in df.columns
