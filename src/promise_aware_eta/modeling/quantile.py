"""Wrappers and helpers for quantile regression and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from promise_aware_eta.modeling.lightgbm_quantile import (
    train_from_config as train_lightgbm_from_config,
)
from promise_aware_eta.modeling.sklearn_quantile import (
    train_hgb_quantile,
    train_linear_quantile,
)


@dataclass
class QuantileModelSpec:
    """Structured specification describing a quantile model training run."""

    name: str
    quantiles: Iterable[float]
    params: Mapping[str, Any] | None = None
    data: Mapping[str, Any] | None = None
    training: Mapping[str, Any] | None = None

    def to_config(self) -> Dict[str, Any]:
        if not self.data:
            raise ValueError("QuantileModelSpec.data must be provided")

        model_cfg: Dict[str, Any] = {"quantiles": list(self.quantiles)}
        if self.params is not None:
            model_cfg["params"] = dict(self.params)

        config: Dict[str, Any] = {
            "model": model_cfg,
            "data": dict(self.data),
        }
        if self.training is not None:
            config["training"] = dict(self.training)
        return config


def train_quantile_model(spec: QuantileModelSpec):
    """Train a quantile model from a :class:`QuantileModelSpec` description."""

    config = spec.to_config()
    model_name = spec.name.lower()

    if model_name == "lightgbm":
        return train_lightgbm_from_config(config)
    if model_name == "linear":
        return train_linear_quantile(config)
    if model_name == "hgb":
        return train_hgb_quantile(config)

    raise ValueError(f"Unsupported quantile model: {spec.name}")
