"""Utilities for capturing environment metadata inside project notebooks."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency in notebook runtimes.
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - IPython not available during tests.
    Markdown = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]


def _run_uv_freeze() -> list[str]:
    """Return the ``uv pip freeze`` output as a list of dependency strings."""

    result = subprocess.run(
        ["uv", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not packages:
        raise RuntimeError("uv pip freeze produced no output; confirm the environment is active.")
    return packages


def capture_environment_metadata(output_path: str | Path | None = None) -> Dict[str, Any]:
    """Capture dependency metadata for embedding in notebooks.

    Parameters
    ----------
    output_path:
        Optional path where the captured metadata should be written as JSON.

    Returns
    -------
    dict
        A dictionary containing the capture timestamp, uv version information,
        and the frozen dependency list. The dictionary is suitable for logging
        in experiment trackers or embedding in notebook markdown cells.
    """

    freeze_packages = _run_uv_freeze()
    uv_version = subprocess.run(
        ["uv", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    metadata: Dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "uv_version": uv_version,
        "dependencies": freeze_packages,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def display_environment_metadata(metadata: Dict[str, Any]) -> None:
    """Render the captured metadata as markdown within a notebook cell."""

    if Markdown is None or display is None:  # pragma: no cover - requires IPython.
        raise RuntimeError("IPython display utilities are unavailable in this environment.")

    header = f"### Environment Snapshot ({metadata['captured_at_utc']})"
    uv_line = f"*uv version*: `{metadata['uv_version']}`"
    dependencies = "\n".join(f"- `{pkg}`" for pkg in metadata["dependencies"])
    display(Markdown("\n".join([header, uv_line, "", dependencies])))


__all__ = [
    "capture_environment_metadata",
    "display_environment_metadata",
]
