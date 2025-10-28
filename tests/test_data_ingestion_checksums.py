import json
from pathlib import Path

import pytest

from promise_aware_eta.data_ingestion import compute_raw_checksums


def test_compute_raw_checksums_writes_payload(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "file_a.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (raw_dir / "file_b.csv").write_text("c,d\n3,4\n", encoding="utf-8")

    payload = compute_raw_checksums(raw_dir, algorithm="sha256")

    checksum_file = raw_dir / "checksums.json"
    assert checksum_file.exists()

    persisted = json.loads(checksum_file.read_text(encoding="utf-8"))
    assert payload == persisted
    assert set(payload["files"].keys()) == {"file_a.csv", "file_b.csv"}


def test_compute_raw_checksums_rejects_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        compute_raw_checksums(tmp_path)
