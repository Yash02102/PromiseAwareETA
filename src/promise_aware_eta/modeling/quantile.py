"""Wrappers and helpers for quantile regression and calibration."""

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class QuantileModelSpec:
    name: str
    quantiles: Iterable[float]
    params: Dict[str, float]


def train_quantile_model(spec: QuantileModelSpec):
    """Train a quantile model described by ``spec``. Implementation pending."""
    raise NotImplementedError("Model training will be implemented in Phase 3.")
