from pathlib import Path
import importlib.util

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analysis" / "notebook_preamble.py"
spec = importlib.util.spec_from_file_location("analysis.notebook_preamble", MODULE_PATH)
notebook_preamble = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(notebook_preamble)


def test_capture_environment_metadata_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        notebook_preamble,
        "_run_uv_freeze",
        lambda: ["pandas==1.0.0", "numpy==2.0.0"],
    )

    class DummyCompletedProcess:
        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        assert cmd == ["uv", "--version"]
        return DummyCompletedProcess("uv 0.4.0")

    monkeypatch.setattr(notebook_preamble.subprocess, "run", fake_run)

    output_path = tmp_path / "metadata.json"
    metadata = notebook_preamble.capture_environment_metadata(output_path)

    assert metadata["uv_version"] == "uv 0.4.0"
    assert metadata["dependencies"] == ["pandas==1.0.0", "numpy==2.0.0"]
    assert output_path.exists()


def test_display_environment_metadata_renders_markdown(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(notebook_preamble, "Markdown", lambda text: f"MD:{text}")

    def fake_display(payload):
        captured["value"] = payload

    monkeypatch.setattr(notebook_preamble, "display", fake_display)

    metadata = {
        "captured_at_utc": "2025-10-20T00:00:00+00:00",
        "uv_version": "uv 0.4.0",
        "dependencies": ["pandas==1.0.0"],
    }

    notebook_preamble.display_environment_metadata(metadata)

    assert "uv 0.4.0" in captured["value"]


def test_display_environment_metadata_requires_ipython(monkeypatch):
    monkeypatch.setattr(notebook_preamble, "Markdown", None)
    monkeypatch.setattr(notebook_preamble, "display", None)

    with pytest.raises(RuntimeError):
        notebook_preamble.display_environment_metadata(
            {
                "captured_at_utc": "2025-10-20T00:00:00+00:00",
                "uv_version": "uv 0.4.0",
                "dependencies": [],
            }
        )
