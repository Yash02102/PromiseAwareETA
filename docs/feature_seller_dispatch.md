# Seller Dispatch Feature Plan

## Objectives
- Capture seller dispatch behavior (time from order approval to ship/handoff) using leakage-safe windows.
- Provide features that segment sellers by responsiveness and variability.
- Supply diagnostics to monitor drift and the impact on promise policies.

## Inputs
- `orders` table with `order_id`, `customer_id`, `order_approved_at`, and `order_delivered_carrier_date`.
- `order_items` table with `seller_id`, `order_id`, `shipping_limit_date`, and item-level timestamps.
- Optional external dispatch events (future enhancement).

## Feature Families
1. **Latency stats:** median and P90 dispatch lag (carrier pickup minus order approval) over trailing windows (30, 60, 90 days).
2. **Volume indicators:** total fulfilled orders and distinct products in the same window.
3. **Timeliness commitments:** share of orders shipped before `shipping_limit_date`.
4. **Seasonality interactions:** difference between recent 14-day lag and 90-day baseline.

## Leakage Controls
- For every order, compute seller statistics using only records with `order_approved_at < current_order_approved_at`.
- Use rolling windows anchored on `order_approved_at`; no future data from calibration or test folds.
- Maintain per-seller buffers for sparse history; backfill with global medians when counts < 10.

## Output Schema (Draft)
- `seller_id`
- `window_days`
- `dispatch_lag_median`
- `dispatch_lag_p90`
- `orders_count`
- `ship_before_limit_rate`
- `recent_vs_baseline_diff`

## Implementation Notes
- Build incremental aggregations keyed by seller with pre-sorted orders.
- Persist intermediate aggregates to `data/interim/seller_dispatch.parquet` for reuse.
- Add unit tests covering leakage guard rails and sparse-history fallback behavior.
