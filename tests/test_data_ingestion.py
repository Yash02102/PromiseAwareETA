import pytest

from promise_aware_eta.analysis.eda import load_delivery_durations, summarize_durations
from promise_aware_eta.data_ingestion import TABLE_FILENAMES, load_olist_tables



def test_load_olist_tables_reads_expected_csvs(sample_raw_dir):
    tables = load_olist_tables(sample_raw_dir)
    assert set(tables.keys()) == set(TABLE_FILENAMES.keys())
    assert len(tables["orders"]) == 2


def test_load_delivery_durations_returns_expected_columns(sample_raw_dir):
    durations = load_delivery_durations(sample_raw_dir)
    assert list(durations.columns) == [
        "order_id",
        "customer_id",
        "purchase_ts",
        "delivered_ts",
        "estimated_delivery_ts",
        "delivery_lag_days",
        "promise_gap_days",
        "is_late_vs_estimated",
    ]

    order_1_duration = durations.loc[durations["order_id"] == "order_1", "delivery_lag_days"].iloc[0]
    assert pytest.approx(order_1_duration, rel=1e-5) == 4 + 2 / 24

    late_mask = durations.set_index("order_id")["is_late_vs_estimated"]
    assert not bool(late_mask["order_1"])
    assert bool(late_mask["order_2"])


def test_summarize_durations_produces_metrics(sample_raw_dir):
    durations = load_delivery_durations(sample_raw_dir)
    summary = summarize_durations(durations)
    metrics = dict(zip(summary["metric"], summary["value"]))

    assert metrics["orders_count"] == pytest.approx(2.0)
    expected_median = durations["delivery_lag_days"].median()
    assert metrics["delivery_days_p50"] == pytest.approx(expected_median)
    assert metrics["late_rate_vs_estimated"] == pytest.approx(0.5)

