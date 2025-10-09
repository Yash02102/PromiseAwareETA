# Data Directory

## Olist Brazilian E-commerce Dataset
- Source: Kaggle dataset [`olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Files: multiple CSV tables covering orders, items, customers, geolocation, sellers, reviews, payments, and products.
- License: Creative Commons (CC BY 4.0) as distributed on Kaggle; review licensing terms before redistribution.

## Download Instructions
1. Install the Kaggle CLI (`pip install kaggle`) and place your API credentials in `%USERPROFILE%/.kaggle/kaggle.json` (Windows) or `~/.kaggle/kaggle.json`.
2. Activate the project environment (for example `make setup`) and run `make data` or `uv run python -m promise_aware_eta.data_ingestion download`.
3. The command stores the raw archive and extracted CSVs in `data/raw/`.
4. Refresh by rerunning with `--force` (for example `uv run python -m promise_aware_eta.data_ingestion download --force`).

## Integrity Checks
After download, verify the presence of key files such as:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`

Optionally compute SHA256 hashes and record them in a checksum file for reproducibility.

## Update Cadence
The Olist dataset is static; no routine refresh schedule is expected. If additional public delivery datasets are incorporated, document them here with acquisition steps and licensing notes.

## Provenance Log
- 2025-09-21: Downloaded via make data after configuring Kaggle CLI (`kaggle datasets download -d olistbr/brazilian-ecommerce`).
