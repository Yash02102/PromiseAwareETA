# Synthetic Quantile Baseline Experiments (2025-10-11)

## Objective
- Establish an end-to-end experiment workflow using a reproducible synthetic dataset.
- Benchmark three quantile regressors (linear, HistGradientBoosting, LightGBM) under a shared configuration.
- Generate visual and tabular artifacts for comparison and future iteration planning.

## Dataset & Setup
- Generated 1,200 synthetic orders spanning 2017-01-01 to 2018-06-30 via `experiments/run_synthetic_suite.py`.
- Features: distance, dispatch delay, weekend indicator, same-state flag.
- Target: `delivery_lag_days` simulated with structured signal + Gaussian noise, clipped to non-negative values.
- Train/validation split uses 2017 orders for training (835 rows) and 2018-H1 for validation (365 rows) per `configs/experiments/synthetic_quantile.yaml`.
- All runs log to `experiments/logs/results.jsonl` (ignored by git); override with `PROMISE_EXPERIMENT_LOG_DIR` if desired.

## Results
Validation pinball loss (lower is better):

| Model      | Q10 | Q50 | Q90 |
|------------|-----|-----|-----|
| Linear     | 0.37 | 0.82 | 0.36 |
| HGB        | 0.39 | 0.88 | 0.46 |
| LightGBM   | 0.38 | 0.89 | 0.44 |

Generated artifacts (not committed):

- `analysis/synthetic_baseline_results.csv`
- `analysis/figures/synthetic_baseline_pinball.png`

Produce both via `python analysis/plot_synthetic_baseline.py` after running the suite.

## Observations
- The linear baseline achieved the lowest average pinball loss across quantiles, suggesting the synthetic signal remains mostly linear.
- Gradient-boosted models showed slightly higher loss at Q50/Q90, potentially due to limited feature richness and the noise floor.
- LightGBM early-stopped quickly with frequent "no further splits" warnings, indicating shallow trees may suffice for this dataset.

## Next Steps
- Introduce heteroskedastic noise or non-linear feature interactions to create a more challenging benchmark.
- Evaluate calibration quality (coverage) and supplement pinball loss with empirical coverage plots.
- Extend logging to include training duration and resource metrics for richer comparisons.
