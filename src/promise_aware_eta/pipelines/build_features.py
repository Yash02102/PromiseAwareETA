"""Pipelines to build feature matrices and datasets."""

from __future__ import annotations

from pathlib import Path

from promise_aware_eta.analysis.eda import load_delivery_durations
from promise_aware_eta.data_ingestion import RAW_DATA_SUBDIR, load_olist_tables
from promise_aware_eta.features import build_model_features, validate_feature_frame

RAW_DIR = Path(RAW_DATA_SUBDIR)
OUTPUT_DIR = Path("data/processed")
FEATURES_PARQUET = OUTPUT_DIR / "features.parquet"
FEATURE_COLUMNS_TXT = OUTPUT_DIR / "features_columns.txt"


def main(raw_dir: Path = RAW_DIR) -> None:
    tables = load_olist_tables(raw_dir)
    feature_matrix = build_model_features(tables)

    durations = load_delivery_durations(raw_dir)
    durations = durations.set_index("order_id")

    dataset = feature_matrix.join(durations, on="order_id", how="inner")
    dataset = dataset.rename(columns={"purchase_ts": "order_purchase_timestamp"})
    dataset = dataset.dropna(subset=["order_purchase_timestamp", "delivery_lag_days"])

    validate_feature_frame(dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(FEATURES_PARQUET, index=False)

    feature_columns = [
        col
        for col in dataset.columns
        if col
        not in {
            "order_id",
            "order_purchase_timestamp",
            "delivery_lag_days",
            "delivered_ts",
            "estimated_delivery_ts",
            "promise_gap_days",
            "is_late_vs_estimated",
            "customer_id",
        }
    ]

    FEATURE_COLUMNS_TXT.write_text("\n".join(feature_columns), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} rows to {FEATURES_PARQUET} with {len(feature_columns)} feature columns."
    )


if __name__ == "__main__":
    main()
