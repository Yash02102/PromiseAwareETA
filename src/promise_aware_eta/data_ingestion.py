"""Utilities for acquiring and loading the Olist public dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd


RAW_DATA_SUBDIR = "data/raw"
KAGGLE_DATASET_ID = "olistbr/brazilian-ecommerce"
DEFAULT_ARCHIVE_NAME = "olistbr-brazilian-ecommerce.zip"
TABLE_FILENAMES: Mapping[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv",
}


def _resolve_raw_directory(data_dir: Path) -> Path:
    candidate = data_dir
    if candidate.is_file():
        raise ValueError(f"Expected directory, received file: {data_dir}")

    if not candidate.exists():
        candidate = data_dir / RAW_DATA_SUBDIR

    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not locate raw data directory under {data_dir} or {data_dir / RAW_DATA_SUBDIR}"
        )

    return candidate


def load_olist_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load raw Olist CSV tables located under ``data_dir`` or ``data_dir / RAW_DATA_SUBDIR``."""
    raw_dir = _resolve_raw_directory(data_dir)
    tables: Dict[str, pd.DataFrame] = {}

    missing_files = []
    for table_name, filename in TABLE_FILENAMES.items():
        csv_path = raw_dir / filename
        if not csv_path.exists():
            missing_files.append(filename)
            continue
        tables[table_name] = pd.read_csv(csv_path)

    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            f"Missing expected Olist CSV files under {raw_dir}: {missing}. Download the dataset first."
        )

    return tables


def download_olist_dataset(raw_dir: Path, force: bool = False) -> Path:
    """Download the Olist dataset using the Kaggle CLI into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / DEFAULT_ARCHIVE_NAME

    if archive_path.exists() and not force:
        return archive_path

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not available. Install it and place credentials in ~/.kaggle/kaggle.json."
        )

    cmd: Iterable[str] = (
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET_ID,
        "-p",
        str(raw_dir),
    )
    if force:
        cmd = tuple(cmd) + ("--force",)

    subprocess.run(cmd, check=True)
    return archive_path


def extract_olist_archive(archive_path: Path, destination: Path, force: bool = False) -> None:
    """Extract the downloaded archive into ``destination``."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        if not force:
            remaining = [name for name in members if not (destination / name).exists()]
            if not remaining:
                return
        zf.extractall(destination)


def compute_raw_checksums(
    raw_dir: Path,
    *,
    algorithm: str = "sha256",
    output: Path | None = None,
) -> dict[str, object]:
    """Compute checksums for raw CSV files and persist them for reproducibility."""

    resolved_dir = _resolve_raw_directory(Path(raw_dir))
    files = sorted(resolved_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {resolved_dir} to checksum.")

    try:
        hashlib.new(algorithm)
    except ValueError as exc:  # pragma: no cover - defensive path for unsupported hashes.
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    checksums: dict[str, str] = {}
    for csv_path in files:
        digest = hashlib.new(algorithm)
        with csv_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksums[csv_path.name] = digest.hexdigest()

    payload: dict[str, object] = {"algorithm": algorithm, "files": checksums}

    output_path = output or resolved_dir / "checksums.json"
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data ingestion helpers for the Olist dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the Olist dataset via Kaggle CLI.")
    download_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(RAW_DATA_SUBDIR),
        help="Directory where raw Olist files will be stored.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing files.",
    )
    download_parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extracting the zip archive after download.",
    )

    extract_parser = subparsers.add_parser("extract", help="Extract a downloaded Olist archive.")
    extract_parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        default=Path(RAW_DATA_SUBDIR) / DEFAULT_ARCHIVE_NAME,
        help="Path to the Kaggle archive (defaults to raw directory).",
    )
    extract_parser.add_argument(
        "--destination",
        type=Path,
        default=Path(RAW_DATA_SUBDIR),
        help="Directory where CSV files will be extracted.",
    )
    extract_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted files.",
    )

    checksum_parser = subparsers.add_parser(
        "checksum",
        help="Compute and persist checksums for raw CSV files.",
    )
    checksum_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(RAW_DATA_SUBDIR),
        help="Directory containing raw Olist CSV files.",
    )
    checksum_parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm to use (default: sha256).",
    )
    checksum_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the checksum JSON payload.",
    )

    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.command == "download":
        archive = download_olist_dataset(args.raw_dir, force=args.force)
        if not args.skip_extract:
            extract_olist_archive(archive, args.raw_dir, force=args.force)
    elif args.command == "extract":
        extract_olist_archive(args.archive, args.destination, force=args.force)
    elif args.command == "checksum":
        payload = compute_raw_checksums(args.raw_dir, algorithm=args.algorithm, output=args.output)
        print(json.dumps(payload, indent=2))
    else:  # pragma: no cover - argparse enforces valid commands.
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    _main()
