"""Tests for forecasting models, constraints enforcement, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from cwfc.forecasting import (
    enforce_constraints,
    mean_reverting_forecast,
    moving_average_forecast,
    persistence_forecast,
)


@pytest.fixture
def synthetic_train():
    return pd.DataFrame({
        "timestamp": [i * 300 for i in range(50)],
        "active_vm_count": [1000] * 50,
        "aggregate_cpu_utilization": [8.0 + 0.05 * i for i in range(50)],
    })


def test_persistence_forecast(synthetic_train):
    preds = persistence_forecast(synthetic_train, horizon=24, target_col="aggregate_cpu_utilization")
    assert len(preds) == 24
    expected_last = float(synthetic_train["aggregate_cpu_utilization"].iloc[-1])
    assert np.allclose(preds.values, expected_last)


def test_moving_average_forecast(synthetic_train):
    preds = moving_average_forecast(synthetic_train, horizon=12, target_col="aggregate_cpu_utilization", window=5)
    assert len(preds) == 12
    expected_ma = float(synthetic_train["aggregate_cpu_utilization"].iloc[-5:].mean())
    assert np.allclose(preds.values, expected_ma)


def test_mean_reverting_forecast_convergence(synthetic_train):
    preds = mean_reverting_forecast(synthetic_train, horizon=30, target_col="aggregate_cpu_utilization", phi=0.8)
    assert len(preds) == 30
    mu = float(synthetic_train["aggregate_cpu_utilization"].mean())
    # As horizon extends, predictions should get closer to the historical mean
    assert abs(preds.iloc[-1] - mu) < abs(preds.iloc[0] - mu)


def test_enforce_constraints():
    # Active VM count non-negative
    raw_vms = np.array([-5.0, 0.0, 10.0])
    bounded_vms = enforce_constraints(raw_vms, target_col="active_vm_count")
    assert (bounded_vms.values >= 0.0).all()
    assert bounded_vms.iloc[0] == 0.0

    # CPU utilization clipped in [0, 100]
    raw_cpu = np.array([-10.0, 50.0, 120.0])
    bounded_cpu = enforce_constraints(raw_cpu, target_col="aggregate_cpu_utilization")
    assert (bounded_cpu.values >= 0.0).all()
    assert (bounded_cpu.values <= 100.0).all()
    assert bounded_cpu.iloc[0] == 0.0
    assert bounded_cpu.iloc[2] == 100.0


def test_empty_train_raises_error():
    empty_df = pd.DataFrame(columns=["timestamp", "aggregate_cpu_utilization"])
    with pytest.raises(ValueError, match="cannot be empty"):
        persistence_forecast(empty_df, horizon=5, target_col="aggregate_cpu_utilization")
    with pytest.raises(ValueError, match="cannot be empty"):
        mean_reverting_forecast(empty_df, horizon=5, target_col="aggregate_cpu_utilization")
