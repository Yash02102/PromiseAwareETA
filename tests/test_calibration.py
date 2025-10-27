import numpy as np
import pytest

from promise_aware_eta.modeling import (
    SplitConformalQuantileCalibrator,
    compute_coverage_diagnostics,
)


def test_split_conformal_calibrator_recovers_coverage():
    y_calib = np.linspace(-2, 2, num=20)
    base_center = y_calib - 1.0
    lower_calib = base_center - 0.1
    upper_calib = base_center + 0.1

    calibrator = SplitConformalQuantileCalibrator(alpha=0.1)
    calibrator.fit(y_calib, lower_pred=lower_calib, upper_pred=upper_calib)

    assert calibrator.is_fitted
    assert calibrator.conformal_quantile_ == pytest.approx(0.9, abs=1e-8)
    assert calibrator.calibration_diagnostics_ is not None
    assert calibrator.calibration_diagnostics_.coverage >= 0.9

    rng = np.random.default_rng(42)
    y_test = rng.normal(loc=0.0, scale=1.0, size=100)
    test_center = y_test - 1.0
    lower_test = test_center - 0.1
    upper_test = test_center + 0.1
    calibrated = calibrator.predict(lower_pred=lower_test, upper_pred=upper_test)

    diagnostics = compute_coverage_diagnostics(
        y_test,
        calibrated["lower"],
        calibrated["upper"],
        target_coverage=0.9,
    )
    assert diagnostics.coverage == pytest.approx(0.9, abs=0.02)
    assert diagnostics.upper_violation_rate <= 0.12


def test_compute_coverage_diagnostics_reports_rates():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([-1.0, 0.8, 1.5, 2.5])
    upper = np.array([0.5, 1.2, 2.4, 2.8])

    diagnostics = compute_coverage_diagnostics(
        y_true,
        lower,
        upper,
        target_coverage=0.8,
    )

    assert diagnostics.coverage == pytest.approx(0.75)
    assert diagnostics.miscoverage_error == pytest.approx(-0.05)
    assert diagnostics.lower_violation_rate == pytest.approx(0.0)
    assert diagnostics.upper_violation_rate == pytest.approx(0.25)
    assert diagnostics.mean_interval_width == pytest.approx(np.mean(upper - lower))
