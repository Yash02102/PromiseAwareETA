"""Generate a synthetic dataset and run baseline quantile experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from promise_aware_eta.modeling import QuantileModelSpec, train_quantile_model, trainers

DATA_DIR = Path("data/processed")
FEATURES_PATH = DATA_DIR / "features.parquet"
FEATURE_COLUMNS_PATH = DATA_DIR / "features_columns.txt"
CONFIG_PATH = Path("configs/experiments/synthetic_quantile.yaml")

RNG = np.random.default_rng(20240214)


def generate_synthetic_dataset(force: bool = False) -> None:
    """Create a reproducible synthetic dataset for experimentation."""
    if FEATURES_PATH.exists() and FEATURE_COLUMNS_PATH.exists() and not force:
        print("Synthetic dataset already exists; skipping generation.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_train = 900
    n_valid = 300
    total = n_train + n_valid

    start_date = np.datetime64("2017-01-01")
    end_date = np.datetime64("2018-06-30")
    purchase_days = RNG.integers(0, int((end_date - start_date).astype(int)) + 1, size=total)
    purchase_ts = pd.to_datetime(start_date + purchase_days, utc=True)

    seller_ids = RNG.choice([f"seller_{i:03d}" for i in range(10)], size=total)
    regions = ["north", "south", "east", "west"]
    seller_region_map = {
        seller: regions[idx % len(regions)]
        for idx, seller in enumerate(sorted({f"seller_{i:03d}" for i in range(10)}))
    }
    seller_region = np.vectorize(seller_region_map.get)(seller_ids)
    distance_km = RNG.gamma(shape=2.0, scale=20.0, size=total)
    dispatch_delay_days = RNG.gamma(shape=2.0, scale=1.0, size=total)
    weekend_purchase = (purchase_ts.weekday >= 5).astype(int)
    same_state = RNG.integers(0, 2, size=total)

    base_time = 2.5 + 0.12 * distance_km + 1.4 * dispatch_delay_days - 0.8 * same_state
    weekend_penalty = weekend_purchase * RNG.normal(loc=1.2, scale=0.3, size=total)
    noise = RNG.normal(loc=0.0, scale=2.0, size=total)
    delivery_lag = base_time + weekend_penalty + noise
    delivery_lag = np.clip(delivery_lag, a_min=0.5, a_max=None)

    df = pd.DataFrame(
        {
            "order_id": [f"order_{i:05d}" for i in range(total)],
            "seller_id": seller_ids,
            "seller_region": seller_region,
            "order_purchase_timestamp": purchase_ts,
            "distance_km": distance_km,
            "dispatch_delay_days": dispatch_delay_days,
            "weekend_purchase": weekend_purchase,
            "same_state": same_state,
            "delivery_lag_days": delivery_lag,
        }
    )

    feature_columns = [
        "distance_km",
        "dispatch_delay_days",
        "weekend_purchase",
        "same_state",
    ]

    df.to_parquet(FEATURES_PATH, index=False)
    FEATURE_COLUMNS_PATH.write_text("\n".join(feature_columns), encoding="utf-8")
    print(f"Generated synthetic dataset with {len(df)} rows at {FEATURES_PATH}.")


def run_baseline_experiments(force_regen: bool = False, use_dispatcher: bool = False) -> None:
    generate_synthetic_dataset(force=force_regen)
    if use_dispatcher:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        base_params = config["model"].get("params", {})
        for model in ("linear", "hgb", "lightgbm"):
            print(f"\n=== Training {model} quantile model (dispatcher) ===")
            spec = QuantileModelSpec(
                name=model,
                quantiles=config["model"]["quantiles"],
                params=base_params,
                data=config["data"],
                training=config.get("training", {}),
            )
            train_quantile_model(spec)
    else:
        for model in ("linear", "hgb", "lightgbm"):
            print(f"\n=== Training {model} quantile model ===")
            trainers.run_training(CONFIG_PATH, model=model)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate the synthetic dataset even if it already exists.",
    )
    parser.add_argument(
        "--use-dispatcher",
        action="store_true",
        help="Call the in-code dispatcher instead of invoking the CLI trainer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_experiments(
        force_regen=args.force_regen,
        use_dispatcher=args.use_dispatcher,
    )
