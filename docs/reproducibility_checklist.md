# Reproducibility Checklist

This checklist tracks the items required before promoting a modeling milestone to release. Owners should confirm each item and
 link evidence in the project wiki.

## Environment & Dependencies

- [ ] Lock file (`uv.lock`) refreshed after dependency updates.
- [ ] `make setup` executed in a fresh environment to verify install instructions.
- [ ] All notebooks record `uv pip freeze` metadata via `analysis/notebook_preamble.py`.

## Data Integrity

- [ ] `data/README.md` reflects latest raw and processed datasets with timestamps.
- [ ] Checksums for raw data dumps stored in `data/raw/checksums.json`.
- [ ] Feature pipelines validate schema using `promise_aware_eta.features.validate_feature_frame`.

## Modeling & Calibration

- [ ] Experiment configs versioned under `configs/experiments/` with semantic names.
- [ ] Baseline quantile suite rerun and summarized in `docs/experiments/`.
- [ ] Conformal calibration diagnostics stored next to run artifacts (`experiments/logs/`).

## Policy Simulation & Fairness

- [ ] Offline simulator runs scripted via `experiments/run_stage_pipeline.py`.
- [ ] Fairness disparity tables exported for sellers and regions.
- [ ] Policy cost parameters reviewed with product/ops stakeholders.

## Documentation & Reporting

- [ ] Changelog entries prepared for README/TASKS updates.
- [ ] Release candidate memo drafted with links to experiment notebooks and dashboards.
- [ ] QA sign-off captured in `docs/decisions/` with reviewer names and dates.
