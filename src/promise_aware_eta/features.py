"""Feature engineering utilities for Promise-Aware ETA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

FEATURE_SNAPSHOT_VERSION = "2025-09-21"


@dataclass
class FeatureBuilderConfig:
    """Configuration for core feature families."""

    include_basket: bool = True
    include_freight: bool = True
    include_temporal: bool = True
    include_payment: bool = True


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
