"""Feature engineering utilities for Promise-Aware ETA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

FEATURE_SNAPSHOT_VERSION = "2025-10-20"

SELLER_DISPATCH_WINDOWS: Sequence[int] = (30, 60, 90)
SELLER_DISPATCH_RECENT_WINDOW_DAYS = 14
SELLER_DISPATCH_MIN_HISTORY = 10
_SECONDS_IN_DAY = 86_400


@dataclass
class FeatureBuilderConfig:
    """Configuration for core feature families."""

    include_basket: bool = True
    include_freight: bool = True
    include_temporal: bool = True
    include_payment: bool = True
    include_seller_dispatch: bool = True


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _seller_dispatch_feature_columns(windows: Sequence[int]) -> Iterable[str]:
    for window in windows:
        suffix = f"{window}d"
        yield f"seller_dispatch_orders_count_{suffix}"
        yield f"seller_dispatch_lag_median_{suffix}"
        yield f"seller_dispatch_lag_p90_{suffix}"
        yield f"seller_dispatch_ship_before_limit_rate_{suffix}"
    yield "seller_dispatch_recent_vs_90d"


def compute_seller_dispatch_features(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    windows: Sequence[int] = SELLER_DISPATCH_WINDOWS,
    recent_window_days: int = SELLER_DISPATCH_RECENT_WINDOW_DAYS,
    min_history: int = SELLER_DISPATCH_MIN_HISTORY,
) -> pd.DataFrame:
    """Return seller-level dispatch rollups for each order without leakage."""

    required_order_cols = {"order_id", "order_approved_at", "order_delivered_carrier_date"}
    required_item_cols = {"order_id", "seller_id", "order_item_id", "shipping_limit_date"}
    if not required_order_cols.issubset(orders.columns) or not required_item_cols.issubset(
        order_items.columns
    ):
        columns = ["order_id", "seller_id", *list(_seller_dispatch_feature_columns(windows))]
        return pd.DataFrame(columns=columns)

    orders = orders.copy()
    orders["order_approved_at"] = _safe_datetime(orders["order_approved_at"])
    orders["order_delivered_carrier_date"] = _safe_datetime(
        orders["order_delivered_carrier_date"]
    )

    seller_orders = (
        order_items.groupby(["order_id", "seller_id"], as_index=False)
        .agg(
            shipping_limit_date=("shipping_limit_date", "min"),
            order_item_count=("order_item_id", "count"),
        )
        .merge(
            orders[["order_id", "order_approved_at", "order_delivered_carrier_date"]],
            on="order_id",
            how="left",
        )
    )

    seller_orders["shipping_limit_date"] = _safe_datetime(seller_orders["shipping_limit_date"])
    seller_orders = seller_orders.dropna(subset=["order_approved_at"])
    if seller_orders.empty:
        columns = ["order_id", "seller_id", *list(_seller_dispatch_feature_columns(windows))]
        return pd.DataFrame(columns=columns)

    seller_orders["dispatch_lag_days"] = (
        (
            seller_orders["order_delivered_carrier_date"]
            - seller_orders["order_approved_at"]
        ).dt.total_seconds()
        / _SECONDS_IN_DAY
    )
    seller_orders["ship_before_limit"] = (
        seller_orders["order_delivered_carrier_date"]
        <= seller_orders["shipping_limit_date"]
    )
    seller_orders.loc[
        seller_orders["ship_before_limit"].isna(), "ship_before_limit"
    ] = False

    dispatch_series = seller_orders["dispatch_lag_days"].dropna()
    global_median = float(dispatch_series.median()) if not dispatch_series.empty else 0.0
    global_p90 = (
        float(dispatch_series.quantile(0.9))
        if len(dispatch_series) > 0
        else global_median
    )
    ship_rate_series = seller_orders["ship_before_limit"].astype(float)
    global_ship_rate = (
        float(ship_rate_series.mean()) if not ship_rate_series.empty else 0.0
    )

    feature_frames = []
    recent_window = f"{recent_window_days}D"
    for seller_id, group in seller_orders.groupby("seller_id", sort=False):
        group = group.sort_values("order_approved_at").set_index("order_approved_at")
        features = pd.DataFrame(index=group.index)
        features["order_id"] = group["order_id"].values
        features["seller_id"] = seller_id

        dispatch_values = group["dispatch_lag_days"]
        ship_values = group["ship_before_limit"].astype(float)

        baseline_series = None
        for window in windows:
            window_key = f"{window}D"
            count_col = f"seller_dispatch_orders_count_{window}d"
            median_col = f"seller_dispatch_lag_median_{window}d"
            p90_col = f"seller_dispatch_lag_p90_{window}d"
            rate_col = f"seller_dispatch_ship_before_limit_rate_{window}d"

            rolling_dispatch = dispatch_values.rolling(window_key, closed="left")
            rolling_ship = ship_values.rolling(window_key, closed="left")

            counts = rolling_dispatch.count()
            medians = rolling_dispatch.median()
            p90s = rolling_dispatch.quantile(0.9)
            ship_rates = rolling_ship.mean()

            features[count_col] = counts.values.astype(float)
            features[median_col] = medians.values
            features[p90_col] = p90s.values
            features[rate_col] = ship_rates.values

            insufficient_history = features[count_col] < float(min_history)
            features.loc[insufficient_history, median_col] = global_median
            features.loc[insufficient_history, p90_col] = global_p90
            features.loc[insufficient_history, rate_col] = global_ship_rate
            features[count_col] = features[count_col].fillna(0.0)

            if window == max(windows):
                baseline_series = features[median_col].copy()

        recent_roll = dispatch_values.rolling(recent_window, closed="left")
        recent_counts = recent_roll.count()
        recent_medians = recent_roll.median()
        recent_series = recent_medians.values
        if baseline_series is None:
            baseline_series = np.full_like(recent_series, global_median, dtype=float)
        insufficient_recent = recent_counts.values < float(min_history)
        recent_series = np.where(insufficient_recent, global_median, recent_series)
        baseline_array = baseline_series.values if hasattr(baseline_series, "values") else baseline_series
        baseline_array = np.where(np.isnan(baseline_array), global_median, baseline_array)
        features["seller_dispatch_recent_vs_90d"] = recent_series - baseline_array

        numeric_cols = [
            col
            for col in features.columns
            if col not in {"order_id", "seller_id"}
        ]
        features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="coerce")
        feature_frames.append(features.reset_index(drop=True))

    result = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    if result.empty:
        columns = ["order_id", "seller_id", *list(_seller_dispatch_feature_columns(windows))]
        return pd.DataFrame(columns=columns)

    numeric_cols = [col for col in result.columns if col not in {"order_id", "seller_id"}]
    result[numeric_cols] = result[numeric_cols].fillna(0.0).astype(float)
    return result


def build_model_features(
    tables: Dict[str, pd.DataFrame],
    config: FeatureBuilderConfig | None = None,
) -> pd.DataFrame:
    """Return a purchase-time feature matrix built from raw Olist tables."""
    cfg = config or FeatureBuilderConfig()
    orders = tables["orders"].copy()
    order_items = tables.get("order_items")
    payments = tables.get("order_payments")

    features = pd.DataFrame(index=orders["order_id"])
    features.index.name = "order_id"

    if cfg.include_basket and order_items is not None:
        basket = (
            order_items.groupby("order_id").agg(
                basket_items=("order_item_id", "count"),
                basket_sellers=("seller_id", "nunique"),
                basket_products=("product_id", "nunique"),
                basket_price_sum=("price", "sum"),
                basket_freight_sum=("freight_value", "sum"),
            )
        )
        features = features.join(basket, how="left")

    if cfg.include_payment and payments is not None:
        payment = (
            payments.groupby("order_id").agg(
                payment_installments_max=("payment_installments", "max"),
                payment_types=("payment_type", "nunique"),
                payment_value_sum=("payment_value", "sum"),
            )
        )
        features = features.join(payment, how="left")

    if cfg.include_seller_dispatch and order_items is not None:
        seller_dispatch = compute_seller_dispatch_features(orders, order_items)
        if not seller_dispatch.empty:
            numeric_cols = [
                col for col in seller_dispatch.columns if col not in {"order_id", "seller_id"}
            ]
            agg_map = {
                col: ("sum" if col.startswith("seller_dispatch_orders_count_") else "mean")
                for col in numeric_cols
            }
            order_level_dispatch = (
                seller_dispatch.groupby("order_id")[numeric_cols].agg(agg_map).fillna(0.0)
            )
            order_level_dispatch = order_level_dispatch.astype(float)
            features = features.join(order_level_dispatch, how="left")

    if cfg.include_temporal:
        orders["order_purchase_timestamp"] = pd.to_datetime(
            orders["order_purchase_timestamp"], utc=True
        )
        temporal = orders.set_index("order_id")["order_purchase_timestamp"].to_frame()
        temporal["purchase_hour"] = temporal["order_purchase_timestamp"].dt.hour
        temporal["purchase_dow"] = temporal["order_purchase_timestamp"].dt.dayofweek
        temporal["purchase_week"] = (
            temporal["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
        )
        temporal["purchase_month"] = temporal["order_purchase_timestamp"].dt.month
        features = features.join(
            temporal.drop(columns=["order_purchase_timestamp"]), how="left"
        )

    features = features.fillna(0)

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].astype(float)

    features = features.reset_index()
    return features
