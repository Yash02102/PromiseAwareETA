"""Delivery promise policy simulation and optimization helpers."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class PromisePolicy:
    quantile: float
    cost_fn: Callable[[float, float], float]


def evaluate_policy(policy: PromisePolicy):
    """Evaluate a promise policy against simulated delivery outcomes."""
    raise NotImplementedError("Policy evaluation will be added alongside the simulator.")
