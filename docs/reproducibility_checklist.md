# Reproducibility Checklist

This checklist tracks the items required before promoting a modeling milestone to release. Owners should confirm each item and
 link evidence in the project wiki.

## Environment & Dependencies

- [x] Lock file (`uv.lock`) refreshed after dependency updates (see `uv lock` run on 2025-10-20).
- [x] `make setup` executed in a fresh environment to verify install instructions (`make setup`, 2025-10-20).
- [x] All notebooks record `uv pip freeze` metadata via `analysis/notebook_preamble.py` (`capture_environment_metadata`).

## Data Integrity

- [x] `data/README.md` reflects latest raw and processed datasets with timestamps (updated 2025-10-20).
- [x] Checksums for raw data dumps stored in `data/raw/checksums.json` (SHA256 snapshot written 2025-10-20).
- [x] Feature pipelines validate schema using `promise_aware_eta.features.validate_feature_frame` (enforced in `pipelines/build_features.py`).

## Modeling & Calibration

- [x] Experiment configs versioned under `configs/experiments/` with semantic names (LightGBM/Linear/HGB quantile configs).
- [x] Baseline quantile suite rerun and summarized in `docs/experiments/` (see calibration + policy memo dated 2025-10-20).
- [x] Conformal calibration diagnostics stored next to run artifacts (`experiments/logs/`, Stage 2 diagnostics CSV + JSON).

## Policy Simulation & Fairness

- [x] Offline simulator runs scripted via `experiments/run_stage_pipeline.py` (stages 1-4 automated, Markdown export).
- [x] Fairness disparity tables exported for sellers and regions (CSV + Markdown under `experiments/logs/`).
- [x] Policy cost parameters reviewed with product/ops stakeholders (documented in release memo and policy simulator guide).

## Documentation & Reporting

- [x] Changelog entries prepared for README/TASKS updates (final summary recorded in `docs/project_plan.md`).
- [x] Release candidate memo drafted with links to experiment notebooks and dashboards (`docs/experiments/2025-10-21_release_candidate_memo.md`).
- [x] QA sign-off captured in `docs/decisions/` with reviewer names and dates (2025-10-21 sign-off note).
