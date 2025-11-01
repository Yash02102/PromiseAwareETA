# Olist Final Run Summary (2025-10-23)

## Overview
- **Dataset**: Olist Brazilian e-commerce orders (train: 2017-01-01–2017-12-31, validation: 2018-01-01–2018-06-30).
- **Model**: LightGBM quantile regression (quantiles 0.1/0.5/0.9) with split-conformal calibration.
- **Execution**: `uv run python -m experiments.run_stage_pipeline --config configs/experiments/quantile_baseline_lightgbm.yaml --model lightgbm`.
- **Artifacts**: Stage logs written to `experiments/logs/` (calibration, policy metrics, regional and seller fairness exports).

## Stage 2 – Calibration Diagnostics
- Observed coverage: **89.97%** vs. 90% target (miscoverage −0.00028).
- Lower / upper violation rates: **0.27% / 9.75%**.
- Mean prediction interval width: **22.03 days**.

_Source: `experiments/logs/stage2_calibration_diagnostics.json`_

## Stage 3 – Offline Policy Simulation
| Policy | Coverage | Avg. Promised (days) | Avg. Actual (days) | Expected Cost |
| --- | --- | --- | --- | --- |
| q50 | 0.511 | 11.02 | 13.33 | 22.35 |
| q90 | **0.842** | **19.98** | 13.33 | **15.44** |

- q90 remains the preferred promise quantile: coverage within 5.8pp of target and ~31% lower asymmetric cost than the median policy.

_Source: `experiments/logs/stage3_policy_metrics.csv`_

## Stage 4 – Fairness Snapshot
- Regional coverage ranges from **60.0% (North)** to **87.3% (Central-West)**.
- Expected cost is lowest in the North (11.62 days) and highest in the Northeast (16.73 days).
- Only **five** validation orders fall in the North region; small sample is driving volatility.

| Region | Coverage | Δ Coverage | Expected Cost | Cost Ratio |
| --- | --- | --- | --- | --- |
| Central-West | 0.873 | 0.000 | 14.687 | 0.878 |
| South | 0.850 | −0.023 | 15.369 | 0.919 |
| Northeast | 0.848 | −0.025 | 16.728 | 1.000 |
| Southeast | 0.841 | −0.032 | 15.443 | 0.923 |
| North | 0.600 | −0.273 | 11.615 | 0.694 |

_Sources: `experiments/logs/stage4_region_fairness.csv`, `experiments/logs/stage4_region_fairness.md`, validation slice counts derived from `data/processed/features.parquet`._

## Next Steps
- Investigate data sparsity for Northern-region sellers; consider stratified resampling or reliability thresholds before reporting fairness deltas.
- Incorporate the new `--config` / `--model` flags into automation so production runs target the LightGBM baseline by default.
