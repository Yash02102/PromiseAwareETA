# Promise-Aware ETA — Final Project Report

## Overview
The Promise-Aware ETA initiative delivers a reproducible workflow for forecasting delivery-time quantiles, calibrating promise intervals, and evaluating downstream policy trade-offs with fairness guardrails. All checklist tasks have been executed and signed off as of 2025-10-21.

## Deliverables
- **Data Pipeline**: Kaggle acquisition tooling, raw checksum ledger (`data/raw/checksums.json`), and schema-validated feature builder (`make features`).
- **Modeling Suite**: Quantile regression baselines (linear, HGB, LightGBM) with split-conformal calibration diagnostics in `experiments/logs/stage2_calibration_diagnostics.json`.
- **Policy Simulator**: Offline evaluation harness (`experiments/run_stage_pipeline.py`) producing cost, coverage, and fairness outputs for policy selection.
- **Fairness Reporting**: Automated Markdown export (`experiments/logs/stage4_region_fairness.md`) and seller/region CSV summaries for notebooks and memos.
- **Documentation**: Release candidate memo, QA sign-off, updated reproducibility checklist, and project plan cross-links.

## Key Results
- Calibrated quantile 0.9 policy achieves **88.2% coverage** with **3.19 days** expected asymmetric cost on validation data.
- Regional coverage ranges **84%–93%** with cost ratios ≤1.0 relative to the south reference segment.
- Seller dispatch features provide leakage-safe rolling statistics with minimum-history backstops, enabling robust fairness analysis.

## Reproducibility Checklist
- Environment setup verified via `make setup` and `analysis/notebook_preamble.capture_environment_metadata` for notebooks.
- Data assets timestamped and checksummed; provenance log updated through 2025-10-20.
- Modeling and policy artifacts versioned under `configs/experiments/` with semantic names.
- QA approval recorded in `docs/decisions/2025-10-21_qa_signoff.md`.

## Next Steps
- Address pandas `groupby.apply` deprecation warning (tracked follow-up issue #45).
- Extend fairness automation to seller segments with bootstrap confidence intervals.
- Prepare thesis chapter integration using artifacts captured in `docs/experiments/2025-10-21_release_candidate_memo.md`.
