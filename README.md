# PromiseAwareETA

Promise-Aware ETA and Delivery Promise Tuning built on public datasets such as Olist.

## Overview
This research project quantifies delivery-time uncertainty per order and optimizes promise policies that balance reliability and speed. We focus on calibrated quantile models, conformal prediction, and offline policy simulation to evaluate trade-offs in late promise rate and customer experience.

## Repository Layout
- `docs/` project plans, research notes, and decision logs.
- `src/promise_aware_eta/` Python package skeleton for data, features, modeling, and policy utilities.
- `data/` placeholders for raw, interim, and processed datasets (tracked via `.gitkeep`).
- `analysis/` exploratory analysis scripts and generated reports.
- `experiments/` scripts, results, and metadata for modeling runs.
- `configs/` configuration files for experiments and pipelines.

## Key Documents
- `docs/project_plan.md` end-to-end roadmap, milestones, and backlog snapshot.
- `TASKS.md` actionable task board grouped by horizon.
- `docs/feature_seller_dispatch.md` seller dispatch feature blueprint.
- `docs/experiment_tracking.md` tracker evaluation and defaults.

## Getting Started

### Environment Setup
- Install `uv` via https://docs.astral.sh/uv/getting-started/ if not available.
- Create an isolated environment: `uv venv`.
- Activate the environment (shell-specific) and install deps: `uv pip install -e .[dev]`.
- Optional: sync to lockfile once created: `uv pip sync pyproject.toml`.

### Helpful Commands
- `make setup` create or update the uv environment.
- `make data` download and extract the Olist dataset via Kaggle (credentials required).
- `make features` build the processed feature matrix (`data/processed/features.parquet`).
- make train-lightgbm train the quantile LightGBM baseline using the latest features.\n- make train-linear run the linear quantile regression baseline.\n- make train-hgb run the HistGradientBoosting quantile baseline.
- `make lint` run Ruff static checks.
- `make test` execute pytest.
- `make eda` run the CLI exploratory summary (writes `analysis/eda_summary.csv`).

1. Install and use `uv` for environment management (see Environment Setup).
2. Download the Olist dataset into `data/raw/` and document the source in `data/README.md`.
3. Run an exploratory analysis script to profile delivery durations and feature distributions.
4. Set up linting, formatting, and testing workflow prior to committing substantive code.

## Status
Early planning phase with repository scaffolding and roadmap in place.

