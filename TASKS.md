# Task Board

## Ready (next up)
- [x] Adopt uv for environment management and document setup commands.
- [x] Download Olist dataset into data/raw/ and record provenance (date: 2025-09-21).
- [x] Create `data/README.md` describing dataset sources, licenses, and update cadence.

## In Discovery
- [x] Draft exploratory analysis script covering target distribution, seasonality, and corridor distance.
- [x] Specify feature engineering plan for seller dispatch metrics with temporal guards.
- [x] Evaluate experiment tracking options (MLflow vs Weights and Biases) and select default.

## Phases
- [x] Implement quantile LightGBM baseline with temporal split validation.
- [x] Implement split-conformal calibration wrapper and coverage diagnostics.
- [x] Build offline policy simulator for quantile selection and cost evaluation.
- [x] Define fairness metrics and reporting templates for regional and seller disparity.
- [x] Compile reproducibility checklist for final release.

## Experiments (Upcoming)
- [x] Execute Stage 1 baseline quantile suite (LightGBM vs. linear vs. HGB).
- [x] Run Stage 2 calibration sweep with split-conformal and rolling updates.
- [x] Build and validate Stage 3 policy simulator experiments (global vs. segment t).
- [x] Complete Stage 4 robustness/fairness runs and summarize findings.
