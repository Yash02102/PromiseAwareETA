"""Training entrypoints for quantile regression baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from promise_aware_eta.modeling.lightgbm_quantile import train_from_config as train_lightgbm_from_config
from promise_aware_eta.modeling.sklearn_quantile import (
    train_hgb_quantile,
    train_linear_quantile,
)

MODEL_CHOICES = ("lightgbm", "linear", "hgb")


def run_training(config_path: Path, model: Literal["lightgbm", "linear", "hgb"]) -> None:
    """Dispatch to the appropriate trainer based on ``model``."""
    if model == "lightgbm":
        boosters = train_lightgbm_from_config(config_path)
        for quantile, booster in boosters.items():
            print(f"Trained quantile {quantile}: best iteration {booster.best_iteration}")
    elif model == "linear":
        train_linear_quantile(config_path)
    elif model == "hgb":
        train_hgb_quantile(config_path)
    else:
        raise ValueError(f"Unsupported model type: {model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quantile regression baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="lightgbm",
        help="Which quantile model to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config, args.model)
