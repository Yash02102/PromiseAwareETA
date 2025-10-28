# Fairness Metrics and Reporting Template

This template standardizes how we quantify and communicate regional and seller-level fairness outcomes for delivery promise po
licies.

## Core Metrics

All metrics are computed via `promise_aware_eta.policy.evaluate_policy_by_group` using the offline simulator outputs. Grouping
 columns should cover at minimum:

- `seller_id` (individual seller disparity)
- `seller_region` (regional disparity; derive from seller metadata when loading features)

For each group we log:

- **Coverage** – share of orders delivered on or before the promised time.
- **Average lateness** – mean number of days late when a promise is missed.
- **Average earliness** – mean slack for on-time orders.
- **Expected cost** – asymmetric under-/over-promise cost from `PromisePolicy.cost_fn`.
- **Sample count** – weighted order count supporting the metrics.

## Disparity Summary

Generate the compact disparity table with `promise_aware_eta.policy.build_fairness_report`:

1. Use coverage and expected cost as the primary fairness metrics.
2. Pick `seller_region` as the default reference dimension.
3. Choose the highest-performing group (or a business-defined reference group) when computing differences/ratios.
4. Flag any group where coverage < target or cost ratio > 1.1 for follow-up analysis.

## Reporting Template

Embed the following table in experiment logs or memos:

| Group | Coverage | Δ Coverage | Expected Cost | Cost Ratio | Sample Count |
|-------|----------|------------|---------------|------------|--------------|
| Ref   | 0.92     | 0.00       | 3.10          | 1.00       | 4,211        |
| ...   | ...      | ...        | ...           | ...        | ...          |

- Δ Coverage and Cost Ratio come from `build_fairness_report`.
- Highlight cells (e.g., bold/italic) when Δ Coverage < -0.03 or Cost Ratio > 1.15.

## Implementation Checklist

- [x] Offline simulator exposes weighted group metrics per policy.
- [x] Fairness report helper returns tidy tables for dashboards or docs.
- [x] Integrate fairness summaries into automated experiment notebooks (Stage 4 exports Markdown via `experiments/run_stage_pipeline.py`).
