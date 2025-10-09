"""Exploratory analysis script for delivery duration distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from promise_aware_eta.data_ingestion import RAW_DATA_SUBDIR, load_olist_tables


DEFAULT_INPUT_DIR = Path(RAW_DATA_SUBDIR)
DEFAULT_OUTPUT = Path('analysis') / 'eda_summary.csv'
SECONDS_IN_DAY = 86_400


def load_delivery_durations(raw_root: Path) -> pd.DataFrame:
    """Return a DataFrame with delivery duration fields required for EDA."""
    tables = load_olist_tables(raw_root)
    orders = tables["orders"].copy()

    time_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in time_cols:
        if col in orders:
            orders[col] = pd.to_datetime(orders[col], utc=True, errors="coerce")

    delivered = orders[orders["order_delivered_customer_date"].notna()].copy()
    if delivered.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "customer_id",
                "purchase_ts",
                "delivered_ts",
                "estimated_delivery_ts",
                "delivery_lag_days",
                "promise_gap_days",
                "is_late_vs_estimated",
            ]
        )

    delivered = delivered.rename(
        columns={
            "order_purchase_timestamp": "purchase_ts",
            "order_delivered_customer_date": "delivered_ts",
            "order_estimated_delivery_date": "estimated_delivery_ts",
        }
    )

    delivered["delivery_lag_days"] = (
        delivered["delivered_ts"] - delivered["purchase_ts"]
    ).dt.total_seconds() / SECONDS_IN_DAY

    if "estimated_delivery_ts" in delivered:
        delivered["promise_gap_days"] = (
            delivered["estimated_delivery_ts"] - delivered["delivered_ts"]
        ).dt.total_seconds() / SECONDS_IN_DAY
        delivered["is_late_vs_estimated"] = delivered["delivered_ts"] > delivered["estimated_delivery_ts"]
    else:
        delivered["promise_gap_days"] = pd.NA
        delivered["is_late_vs_estimated"] = pd.NA

    columns = [
        "order_id",
        "customer_id",
        "purchase_ts",
        "delivered_ts",
        "estimated_delivery_ts",
        "delivery_lag_days",
        "promise_gap_days",
        "is_late_vs_estimated",
    ]
    return delivered[columns]


def summarize_durations(durations: pd.DataFrame) -> pd.DataFrame:
    """Compute basic summary statistics for delivery durations."""
    if durations.empty:
        return pd.DataFrame([
            {"metric": "orders_count", "value": 0.0},
            {"metric": "delivery_days_mean", "value": float("nan")},
            {"metric": "delivery_days_p50", "value": float("nan")},
            {"metric": "late_rate_vs_estimated", "value": float("nan")},
        ])

    delivery_days = durations["delivery_lag_days"].dropna()
    summary_rows = [
        {"metric": "orders_count", "value": float(len(durations))},
        {"metric": "delivery_days_mean", "value": float(delivery_days.mean())},
        {"metric": "delivery_days_std", "value": float(delivery_days.std(ddof=0))},
    ]

    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        metric_name = f"delivery_days_p{int(quantile * 100)}"
        summary_rows.append({"metric": metric_name, "value": float(delivery_days.quantile(quantile))})

    if "promise_gap_days" in durations:
        promise_gap = durations["promise_gap_days"].dropna()
        if not promise_gap.empty:
            summary_rows.append(
                {
                    "metric": "promise_gap_mean",
                    "value": float(promise_gap.mean()),
                }
            )
            late_mask = durations["is_late_vs_estimated"].fillna(False)
            summary_rows.append(
                {
                    "metric": "late_rate_vs_estimated",
                    "value": float(late_mask.mean()),
                }
            )
        else:
            summary_rows.append({"metric": "promise_gap_mean", "value": float("nan")})
            summary_rows.append({"metric": "late_rate_vs_estimated", "value": float("nan")})
    else:
        summary_rows.append({"metric": "promise_gap_mean", "value": float("nan")})
        summary_rows.append({"metric": "late_rate_vs_estimated", "value": float("nan")})

    return pd.DataFrame(summary_rows)


def main(raw_root: Path, output_path: Optional[Path]) -> None:
    """Run the exploratory analysis pipeline using script-friendly components."""
    durations = load_delivery_durations(raw_root)
    summary = summarize_durations(durations)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline exploratory statistics for delivery durations."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw Olist tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file path to store summary metrics.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="If set, skip writing summary metrics to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = None if args.skip_write else args.output
    main(args.raw_root, output_path)
