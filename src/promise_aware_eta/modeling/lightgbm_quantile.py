"""LightGBM quantile regression helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import lightgbm as lgb
import yaml
from sklearn.metrics import mean_pinball_loss

from promise_aware_eta.modeling.datasets import load_experiment_splits
from promise_aware_eta.experiments.log_utils import log_experiment_results


class QuantileGBMTrainer:
    """Train LightGBM models for multiple quantiles using a shared dataset."""

    def __init__(self, config: Mapping[str, object], *, source_path: Path):
        self.config_path = source_path
        self.config: Dict = dict(config)

    @classmethod
    def from_path(cls, config_path: Path) -> "QuantileGBMTrainer":
        with config_path.open("r", encoding="utf-8") as handle:
            config: Dict = yaml.safe_load(handle)
        return cls(config, source_path=config_path)

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, object], *, source_path: Path | None = None
    ) -> "QuantileGBMTrainer":
        return cls(config, source_path=source_path or Path("in-memory-config.yaml"))

    def load_data(self):
        return load_experiment_splits(self.config)

    def train(self) -> Dict[float, lgb.Booster]:
        X_train, y_train, X_valid, y_valid, feature_cols = self.load_data()
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        boosters: Dict[float, lgb.Booster] = {}
        quantiles: Iterable[float] = self.config["model"]["quantiles"]
        params_cfg = self.config["model"].get("params", {})
        if isinstance(params_cfg, dict) and "lightgbm" in params_cfg:
            params_base = params_cfg["lightgbm"].copy()
        else:
            params_base = params_cfg.copy()
        training_cfg = self.config.get("training", {})
        should_report = bool(training_cfg.get("report_metrics", True))
        metrics = []

        for quantile in quantiles:
            params = params_base.copy()
            params["alpha"] = quantile

            callbacks = []
            early_stop = training_cfg.get("early_stopping_rounds")
            if early_stop:
                callbacks.append(lgb.early_stopping(int(early_stop), verbose=False))

            booster = lgb.train(
                params,
                train_data,
                num_boost_round=int(training_cfg.get("num_boost_round", 1000)),
                valid_sets=[valid_data],
                callbacks=callbacks if callbacks else None,
            )
            boosters[quantile] = booster

            loss = float("nan")
            if should_report:
                valid_pred = booster.predict(X_valid)
                loss = float(mean_pinball_loss(y_valid, valid_pred, alpha=quantile))
                print(
                    f"Quantile {quantile}: validation pinball loss {loss:.4f}"
                )
            metrics.append({"quantile": float(quantile), "pinball_loss": loss})

        log_experiment_results(
            model="lightgbm",
            config_path=self.config_path,
            metrics=metrics,
            train_rows=len(X_train),
            valid_rows=len(X_valid),
            feature_columns=feature_cols,
        )
        return boosters


ConfigInput = Union[Path, str, os.PathLike[str], Mapping[str, object]]


def train_from_config(config: ConfigInput) -> Dict[float, lgb.Booster]:
    """Train LightGBM quantile models from a YAML config path or mapping."""
    if isinstance(config, Path):
        trainer = QuantileGBMTrainer.from_path(config)
    elif isinstance(config, (str, os.PathLike)):
        trainer = QuantileGBMTrainer.from_path(Path(config))
    else:
        trainer = QuantileGBMTrainer.from_mapping(config)
    return trainer.train()
