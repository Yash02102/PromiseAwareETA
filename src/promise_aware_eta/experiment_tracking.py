"""Configuration and selection logic for experiment tracking tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrackerChoice = Literal["mlflow", "wandb", "none"]


@dataclass
class TrackerConfig:
    """Minimal tracker configuration toggled at runtime."""

    choice: TrackerChoice
    tracking_uri: str | None = None
    experiment_name: str | None = None


DEFAULT_TRACKER = TrackerConfig(choice="mlflow", tracking_uri="mlruns", experiment_name="baseline")


def get_tracker_config(choice: TrackerChoice = "mlflow") -> TrackerConfig:
    """Return tracker configuration for the requested backend."""
    if choice == "mlflow":
        return TrackerConfig(choice="mlflow", tracking_uri="mlruns", experiment_name="promise_eta")
    if choice == "wandb":
        return TrackerConfig(choice="wandb", tracking_uri=None, experiment_name="promise_eta")
    return TrackerConfig(choice="none")
