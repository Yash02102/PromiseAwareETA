"""Experiment logging helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

LOG_DIR_ENV = "PROMISE_EXPERIMENT_LOG_DIR"
DEFAULT_LOG_DIR = Path("experiments/logs")


def log_experiment_results(
    *,
    model: str,
    config_path: Path,
    metrics: Iterable[Mapping[str, float]],
    train_rows: int,
    valid_rows: int,
    feature_columns: Iterable[str],
) -> Path:
    """Append experiment metrics to a JSONL log file."""
    log_dir = Path(os.getenv(LOG_DIR_ENV, DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "results.jsonl"

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "model": model,
        "config_path": str(config_path),
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "feature_count": len(list(feature_columns)),
        "metrics": list(metrics),
    }

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    return log_file
