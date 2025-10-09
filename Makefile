.PHONY: setup lint format test eda data features train-lightgbm train-linear train-hgb

setup:
	uv venv
	uv pip install -e .[dev]

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

data:
	uv run python -m promise_aware_eta.data_ingestion download

eda:
	uv run python -m promise_aware_eta.analysis.eda --raw-root data/raw

features:
	uv run python -m promise_aware_eta.pipelines.build_features

train-lightgbm:
	uv run python -m promise_aware_eta.modeling.trainers --model lightgbm --config configs/experiments/quantile_baseline_lightgbm.yaml

train-linear:
	uv run python -m promise_aware_eta.modeling.trainers --model linear --config configs/experiments/quantile_baseline_linear.yaml

train-hgb:
	uv run python -m promise_aware_eta.modeling.trainers --model hgb --config configs/experiments/quantile_baseline_hgb.yaml
