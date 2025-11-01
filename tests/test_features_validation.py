import pandas as pd
import pytest

from promise_aware_eta.features import validate_feature_frame


def _sample_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                utc=True,
            ),
            "delivery_lag_days": [3.5, 4.0],
            "feature_a": [0.1, 0.2],
            "feature_b": [1.0, 2.0],
        }
    )


def test_validate_feature_frame_passes_for_clean_frame():
    frame = _sample_feature_frame()
    result = validate_feature_frame(frame)
    pd.testing.assert_frame_equal(result, frame)


@pytest.mark.parametrize(
    "mutator, expected_exception",
    [
        (lambda df: df.assign(order_id=["o1", "o1"]), ValueError),
        (lambda df: df.drop(columns=["order_purchase_timestamp"]), ValueError),
        (lambda df: df.assign(order_purchase_timestamp=pd.to_datetime(["2025-01-01", "2025-01-02"])), TypeError),
        (lambda df: df.assign(delivery_lag_days=[3.0, float("nan")]), ValueError),
        (lambda df: df.assign(feature_a=["x", "y"]), TypeError),
    ],
)
def test_validate_feature_frame_raises_on_invalid_frames(mutator, expected_exception):
    frame = _sample_feature_frame()
    bad_frame = mutator(frame.copy())
    with pytest.raises(expected_exception):
        validate_feature_frame(bad_frame)
