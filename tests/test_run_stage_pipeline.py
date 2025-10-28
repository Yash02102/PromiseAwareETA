from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

pkg_spec = importlib.util.spec_from_file_location("experiments", EXPERIMENTS_DIR / "__init__.py")
experiments_pkg = importlib.util.module_from_spec(pkg_spec)
assert pkg_spec.loader is not None
pkg_spec.loader.exec_module(experiments_pkg)
sys.modules.setdefault("experiments", experiments_pkg)

synthetic_spec = importlib.util.spec_from_file_location(
    "experiments.run_synthetic_suite",
    EXPERIMENTS_DIR / "run_synthetic_suite.py",
)
synthetic_module = importlib.util.module_from_spec(synthetic_spec)
assert synthetic_spec.loader is not None
synthetic_spec.loader.exec_module(synthetic_module)
sys.modules["experiments.run_synthetic_suite"] = synthetic_module

MODULE_PATH = EXPERIMENTS_DIR / "run_stage_pipeline.py"
spec = importlib.util.spec_from_file_location("experiments.run_stage_pipeline", MODULE_PATH)
run_stage_pipeline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_stage_pipeline)


def test_fairness_report_markdown_formats_table():
    report = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "metric": ["coverage", "coverage", "expected_cost", "expected_cost"],
            "value": [0.95, 0.90, 2.1, 2.4],
            "difference_from_reference": [0.0, -0.05, 0.0, 0.3],
            "ratio_to_reference": [1.0, 0.947368, 1.0, 1.142857],
        }
    )

    markdown = run_stage_pipeline._fairness_report_markdown(report)

    assert "Group" in markdown
    assert markdown.count("\n") >= 2


def test_fairness_report_markdown_requires_metrics():
    incomplete = pd.DataFrame(
        {
            "group": ["A"],
            "metric": ["coverage"],
            "value": [0.9],
            "difference_from_reference": [0.0],
            "ratio_to_reference": [1.0],
        }
    )

    with pytest.raises(ValueError):
        run_stage_pipeline._fairness_report_markdown(incomplete)
