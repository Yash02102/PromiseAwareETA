"""Summarize and visualize synthetic baseline experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LOG_PATH = Path("experiments/logs/results.jsonl")
DEFAULT_TABLE = Path("analysis/synthetic_baseline_results.csv")
DEFAULT_FIG = Path("analysis/figures/synthetic_baseline_pinball.png")
DEFAULT_CONFIG = Path("configs/experiments/synthetic_quantile.yaml")


def load_results(log_path: Path, config_path: str | Path) -> pd.DataFrame:
    records: list[dict] = []
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    target_config = str(config_path)

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("config_path") != target_config:
                continue
            model = payload["model"]
            train_rows = payload.get("train_rows", 0)
            valid_rows = payload.get("valid_rows", 0)
            for metric in payload.get("metrics", []):
                records.append(
                    {
                        "model": model,
                        "quantile": metric.get("quantile"),
                        "pinball_loss": metric.get("pinball_loss"),
                        "train_rows": train_rows,
                        "valid_rows": valid_rows,
                        "timestamp": payload.get("timestamp"),
                    }
                )
    if not records:
        raise ValueError(f"No entries found for config {config_path}")
    return pd.DataFrame.from_records(records)


def plot_pinball(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="quantile", columns="model", values="pinball_loss")
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Pinball Loss (Validation)")
    ax.set_xlabel("Quantile")
    ax.set_title("Synthetic Baseline Pinball Loss by Model")
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the experiment JSONL log produced by the training scripts.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Only include log entries that match this experiment config path.",
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=DEFAULT_TABLE,
        help="Destination CSV path for the summarized metrics table.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIG,
        help="Destination image path for the pinball loss comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_results(args.log_path, args.config_path)
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.table_path, index=False)
    plot_pinball(df, args.figure_path)
    print(f"Saved table to {args.table_path}")
    print(f"Saved figure to {args.figure_path}")


if __name__ == "__main__":
    main()
