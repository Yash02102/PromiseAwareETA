# Seller Dispatch Feature Rollups

## Overview
The feature builder now materializes rolling seller dispatch aggregates that capture how quickly each seller hands off packages after an order is approved. These metrics are leakage-safe because every statistic is computed using only historical orders with `order_approved_at` strictly less than the order being scored.

## Generated Columns
All columns are prefixed with `seller_dispatch_` and are aggregated to the order level by averaging across sellers on multi-seller baskets (order counts are summed).

| Column | Description |
| --- | --- |
| `seller_dispatch_orders_count_{30,60,90}d` | Number of historical orders for the seller in the trailing 30/60/90 days (summed across sellers per order). |
| `seller_dispatch_lag_median_{30,60,90}d` | Median dispatch lag (days between `order_approved_at` and `order_delivered_carrier_date`) over the trailing window. |
| `seller_dispatch_lag_p90_{30,60,90}d` | 90th percentile dispatch lag over the trailing window. |
| `seller_dispatch_ship_before_limit_rate_{30,60,90}d` | Share of orders shipped before the `shipping_limit_date` in the trailing window. |
| `seller_dispatch_recent_vs_90d` | Difference between the trailing 14-day median dispatch lag and the 90-day median baseline. |

When the trailing window contains fewer than 10 historical orders, latency and timeliness metrics backfill to global seller medians to ensure stability for sparse-history sellers.

## Configuration
A new flag `FeatureBuilderConfig.include_seller_dispatch` (default: `True`) allows pipelines to enable or disable seller dispatch features. Existing CLI entry points pick up the flag automatically; set it to `False` when instantiating `FeatureBuilderConfig` if you need to skip these features for ablation studies.

## Pipeline Outputs
`pipelines/build_features.py` persists the augmented feature matrix and updates `features_columns.txt` so downstream training artifacts include the seller dispatch fields.
