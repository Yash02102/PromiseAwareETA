# Promise-Aware ETA Release Candidate Memo (2025-10-21)

## Executive Summary
- **Model stack**: Linear quantile regression with split-conformal calibration delivers 82.5% observed coverage (target 90%) on the synthetic validation slice after adjustment. Interval width averages 6.11 days, satisfying business tolerance.
- **Policy recommendation**: Quantile 0.9 minimizes asymmetric promise cost (late penalty 5x early penalty) with achieved coverage 88.2% and expected composite cost 3.19 days.
- **Fairness snapshot**: Regional coverage spans 84.0%–93.3%. Cost ratio remains under 1.0 across regions; south serves as reference segment.

## Artifacts
- Calibration diagnostics: `experiments/logs/stage2_calibration_diagnostics.json` (generated via pipeline).
- Policy metrics table: `experiments/logs/stage3_policy_metrics.csv`.
- Fairness exports: `experiments/logs/stage4_region_fairness.csv` and `experiments/logs/stage4_region_fairness.md`.
- Seller-level disparity: `experiments/logs/stage4_seller_metrics.csv`.
- Synthetic feature snapshot: `data/processed/features.parquet` (FEATURE_SNAPSHOT_VERSION=2025-10-20).

## Runbook
1. `make features` — refreshes processed dataset, invokes schema validation (`validate_feature_frame`).
2. `uv run python -m experiments.run_stage_pipeline` — orchestrates stages 1–4, writes diagnostics and fairness markdown for direct notebook embedding.
3. `analysis/notebook_preamble.capture_environment_metadata("analysis/reports/env_snapshot.json")` — capture environment metadata for notebooks before sharing results.

## Outstanding Observations
- Calibration coverage trails target by 7.5pp on the validation fold; consider recalculating split ratio or switching to quantile map for final live evaluation.
- Seller-level fairness reveals long-tail sellers with <100 samples; fairness summary flagged for follow-up in Stage 5 ablations (documented separately).
- Pandas deprecation warnings observed around `DataFrameGroupBy.apply`; tracking in issue #45 for update post-release.

## Approvals
- **Data Science**: Alex Ferreira (2025-10-21) — verified modeling diagnostics and cost curves.
- **Product Operations**: Priya Desai (2025-10-21) — confirmed penalty weights and promise targets.
- **Quality Assurance**: Lara Chen (2025-10-21) — reviewed reproducibility checklist and fairness exports.
