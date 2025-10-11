import pandas as pd
import pytest

from promise_aware_eta.data_ingestion import load_olist_tables
from promise_aware_eta.features import build_model_features, compute_seller_dispatch_features
from promise_aware_eta.pipelines.build_features import main as build_features_main


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
        "seller_dispatch_orders_count_30d",
        "seller_dispatch_lag_median_30d",
        "seller_dispatch_lag_p90_30d",
        "seller_dispatch_ship_before_limit_rate_30d",
        "seller_dispatch_orders_count_60d",
        "seller_dispatch_lag_median_60d",
        "seller_dispatch_lag_p90_60d",
        "seller_dispatch_ship_before_limit_rate_60d",
        "seller_dispatch_orders_count_90d",
        "seller_dispatch_lag_median_90d",
        "seller_dispatch_lag_p90_90d",
        "seller_dispatch_ship_before_limit_rate_90d",
        "seller_dispatch_recent_vs_90d",
    }
    assert expected_columns.issubset(set(features.columns))


def test_build_features_pipeline_creates_artifacts(tmp_path, monkeypatch, sample_raw_dir):
    output_dir = tmp_path / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PROMISE_FEATURE_RAW_DIR", str(sample_raw_dir))

    # Monkeypatch module-level constants
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

    seller_dispatch_columns = [
        col for col in feature_columns if col.startswith("seller_dispatch_")
    ]
    assert seller_dispatch_columns, "Seller dispatch features should be persisted"

    assert "delivery_lag_days" in df.columns
    assert "order_purchase_timestamp" in df.columns


def test_compute_seller_dispatch_sparse_history_backfills_global_values():
    base_date = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "order_approved_at": [
                base_date,
                base_date + pd.Timedelta(days=1),
                base_date + pd.Timedelta(days=2),
            ],
            "order_delivered_carrier_date": [
                base_date + pd.Timedelta(days=1),
                base_date + pd.Timedelta(days=3),
                base_date + pd.Timedelta(days=12),
            ],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "order_item_id": [1, 1, 1],
            "seller_id": ["seller_a"] * 3,
            "shipping_limit_date": [
                base_date + pd.Timedelta(days=2),
                base_date + pd.Timedelta(days=5),
                base_date + pd.Timedelta(days=1),
            ],
        }
    )

    features = compute_seller_dispatch_features(orders, order_items)
    third_order = features.loc[features["order_id"] == "o3"].iloc[0]

    # Global median of dispatch lag days should backfill due to insufficient history (<10 orders).
    assert pytest.approx(2.0, rel=1e-6) == third_order["seller_dispatch_lag_median_30d"]
    assert pytest.approx(2.0, rel=1e-6) == third_order["seller_dispatch_lag_median_60d"]
    assert pytest.approx(2.0, rel=1e-6) == third_order["seller_dispatch_lag_median_90d"]

    # Ship rate fallback should use global mean (two of three orders shipped before limit).
    expected_ship_rate = pytest.approx(2 / 3, rel=1e-6)
    assert expected_ship_rate == third_order["seller_dispatch_ship_before_limit_rate_30d"]
    assert 0.0 == pytest.approx(third_order["seller_dispatch_recent_vs_90d"], rel=1e-6)


def test_compute_seller_dispatch_leakage_guardrail_respects_ordering():
    base_date = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    order_ids = [f"order_{i}" for i in range(1, 12)]
    approved_times = [base_date + pd.Timedelta(days=i - 1) for i in range(1, 12)]
    dispatch_lags = [float(i) for i in range(1, 11)] + [0.1]

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_approved_at": approved_times,
            "order_delivered_carrier_date": [
                t + pd.Timedelta(days=lag) for t, lag in zip(approved_times, dispatch_lags)
            ],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_item_id": [1] * len(order_ids),
            "seller_id": ["seller_a"] * len(order_ids),
            "shipping_limit_date": [
                t + pd.Timedelta(days=lag + 1)
                for t, lag in zip(approved_times, dispatch_lags)
            ],
        }
    )

    features = compute_seller_dispatch_features(orders, order_items)
    last_order = features.loc[features["order_id"] == "order_11"].iloc[0]

    assert last_order["seller_dispatch_orders_count_90d"] == pytest.approx(10.0)
    assert last_order["seller_dispatch_lag_median_90d"] == pytest.approx(5.5, rel=1e-6)
    assert last_order["seller_dispatch_lag_p90_90d"] == pytest.approx(9.1, rel=1e-6)
    assert last_order["seller_dispatch_recent_vs_90d"] == pytest.approx(0.0, rel=1e-6)
