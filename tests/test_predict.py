"""Tests for prediction generation, schema validation, and smoke workflows."""

import numpy as np
import pandas as pd
import pytest

from cwfc.predict import (
    REQUIRED_PREDICTION_COLUMNS,
    generate_forecasts,
    validate_predictions_df,
    validate_predictions_file,
)


def test_validate_predictions_df_valid():
    timestamps = [33000 + i * 300 for i in range(24)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": [10000] * 24,
        "aggregate_cpu_utilization": [9.35] * 24,
    })
    # Should not raise
    validate_predictions_df(df, expected_timestamps=timestamps, expected_rows=24)


def test_validate_predictions_df_invalid_columns():
    df = pd.DataFrame({
        "timestamp": range(24),
        "active_vm_count": range(24),
        "wrong_cpu_col": range(24),
    })
    with pytest.raises(ValueError, match="Prediction columns must be exactly"):
        validate_predictions_df(df)


def test_validate_predictions_df_wrong_row_count():
    df = pd.DataFrame({
        "timestamp": range(20),
        "active_vm_count": range(20),
        "aggregate_cpu_utilization": [10.0] * 20,
    })
    with pytest.raises(ValueError, match="Expected exactly 24 prediction rows"):
        validate_predictions_df(df, expected_rows=24)


def test_validate_predictions_df_non_finite():
    timestamps = [33000 + i * 300 for i in range(24)]
    cpu_vals = [9.0] * 23 + [np.nan]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": [10000] * 24,
        "aggregate_cpu_utilization": cpu_vals,
    })
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_predictions_df(df)


def test_validate_predictions_df_bounds_violation():
    timestamps = [33000 + i * 300 for i in range(24)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": [10000] * 23 + [-1],
        "aggregate_cpu_utilization": [9.0] * 24,
    })
    with pytest.raises(ValueError, match="Constraint violation: active_vm_count must be >= 0"):
        validate_predictions_df(df)


def test_validate_predictions_file_on_actual_output():
    # Validates the project's generated predictions.csv
    df = validate_predictions_file("predictions.csv", expected_rows=24)
    assert len(df) == 24
    assert list(df.columns) == REQUIRED_PREDICTION_COLUMNS


def test_synthetic_replacement_dataset_smoke(tmp_path):
    """Verify that pipeline functions correctly on a small replacement dataset.

    Reviewers test this to verify that values and timestamps are not hard-coded.
    """
    records = []
    # 40 timestamps, 5 VMs
    for t_idx in range(40):
        t = 10000 + t_idx * 300
        for vm_idx in range(1, 6):
            records.append({
                "timestamp": t,
                "vm_id": f"smoke_vm_{vm_idx}",
                "min_cpu": 1.0,
                "max_cpu": 15.0,
                "avg_cpu": 5.0 + (t_idx % 3),
            })
    smoke_df = pd.DataFrame(records)
    smoke_csv = tmp_path / "vm_cpu_readings.csv"
    smoke_df.to_csv(smoke_csv, index=False)

    preds = generate_forecasts(smoke_csv, train_size=25, holdout_size=15)
    assert len(preds) == 15
    assert list(preds.columns) == REQUIRED_PREDICTION_COLUMNS
    assert preds["timestamp"].iloc[0] == 10000 + 25 * 300
    assert preds["timestamp"].iloc[-1] == 10000 + 39 * 300
    assert (preds["active_vm_count"] == 5).all()
    assert (preds["aggregate_cpu_utilization"] >= 0.0).all()
    assert (preds["aggregate_cpu_utilization"] <= 100.0).all()
