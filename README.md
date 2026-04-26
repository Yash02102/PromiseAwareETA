# PromiseAwareETA

Promise-Aware ETA and Delivery Promise Tuning built on public datasets such as Olist.

## Overview
This research project quantifies delivery-time uncertainty per order and optimizes promise policies that balance reliability and speed. We focus on calibrated quantile models, conformal prediction, and offline policy simulation to evaluate trade-offs in late promise rate and customer experience.

## Repository Layout
- `docs/` project plans, research notes, and decision logs.
- `src/promise_aware_eta/` Python package skeleton for data, features, modeling, and policy utilities.
- `data/` placeholders and provenance notes for externally downloaded datasets.
- `analysis/` exploratory analysis scripts and generated reports.
- `experiments/` scripts, results, and metadata for modeling runs.
- `configs/` configuration files for experiments and pipelines.

## Key Documents
- `docs/project_plan.md` end-to-end roadmap, milestones, and backlog snapshot.
- `TASKS.md` actionable task board grouped by horizon.
- `docs/feature_seller_dispatch.md` seller dispatch feature blueprint.
- `docs/experiment_tracking.md` tracker evaluation and defaults.
- `docs/policy_simulator.md` staged offline simulation workflow and artifacts.
- `docs/fairness_reporting.md` fairness metrics and reporting template.
- `docs/reproducibility_checklist.md` release readiness checklist.

## Getting Started

### Environment Setup
- Install `uv` via https://docs.astral.sh/uv/getting-started/ if not available.
- Create an isolated environment: `uv venv`.
- Activate the environment (shell-specific) and install deps: `uv pip install -e .[dev]`.
- Optional: sync to the checked-in lockfile: `uv sync --extra dev`.

### Helpful Commands
- `make setup` create or update the uv environment.
- `make data` download and extract the Olist dataset from Kaggle (credentials required).
- `make features` build the processed feature matrix (`data/processed/features.parquet`).
- `make train-lightgbm` train the quantile LightGBM baseline using the latest features.
- `make train-linear` run the linear quantile regression baseline.
- `make train-hgb` run the HistGradientBoosting quantile baseline.
- `make lint` run Ruff static checks.
- `make test` execute pytest.
- `make eda` run the CLI exploratory summary (writes `analysis/eda_summary.csv`).

1. Install and use `uv` for environment management (see Environment Setup).
2. Download the Olist dataset into `data/raw/` from Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce.
3. Run an exploratory analysis script to profile delivery durations and feature distributions.
4. Set up linting, formatting, and testing workflow prior to committing substantive code.

## Data and License Notes
The raw Olist CSV files are intentionally not committed. Download them from Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce.

Project source code is released under the MIT License. Dataset files and third-party references remain governed by their original licenses.

## Status
Research pipeline with reproducible experiments, generated reports, and test coverage for the core data, feature, modeling, and policy utilities.

