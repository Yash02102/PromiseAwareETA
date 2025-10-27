# Offline Policy Simulator

The offline simulator evaluates delivery promise policies against realized delivery lags using quantile model predictions. It
 enables staged experimentation without deploying models to production.

## Workflow

1. **Generate dataset** – `experiments/run_synthetic_suite.py` creates a reproducible synthetic corpus with seller and regional
   metadata.
2. **Train baselines** – Stage 1 of `experiments/run_stage_pipeline.py` trains quantile regressors via the shared dispatcher.
3. **Calibrate quantiles** – Stage 2 applies split-conformal calibration to tighten coverage guarantees.
4. **Simulate promises** – Stage 3 evaluates multiple quantile-based policies, logging coverage, lateness, and cost metrics.
5. **Audit fairness** – Stage 4 computes seller and region disparity tables for the selected policy.

## Usage

```bash
python experiments/run_stage_pipeline.py --force-regen
```

Artifacts land in `experiments/logs/`:

- `stage3_policy_metrics.csv`
- `stage4_seller_metrics.csv`
- `stage4_region_fairness.csv`

Use these outputs to populate experiment notebooks and fairness reports. The simulator relies on `promise_aware_eta.policy`
helpers for metric computation and integrates directly with the conformal calibration utilities in `promise_aware_eta.modeling`.
