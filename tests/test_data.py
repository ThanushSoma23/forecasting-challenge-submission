"""Tests for data validation, aggregation, and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from cwfc.data import (
    build_aggregate_targets,
    check_timestamp_regularity,
    load_cpu_readings,
    split_train_holdout,
)


@pytest.fixture
def synthetic_readings():
    """Create a minimal synthetic readings dataset for fast testing."""
    records = []
    # 3 timestamps: 0, 300, 600. 4 VMs: vm_1, vm_2, vm_3, vm_4
    for t in [0, 300, 600]:
        for i in range(1, 5):
            records.append({
                "timestamp": t,
                "vm_id": f"vm_{i}",
                "min_cpu": float(i * 2),
                "max_cpu": float(i * 10),
                "avg_cpu": float(i * 5),
            })
    return pd.DataFrame(records)


def test_build_aggregate_targets(synthetic_readings):
    agg = build_aggregate_targets(synthetic_readings)

    assert list(agg.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(agg) == 3
    assert np.array_equal(agg["timestamp"].values, [0, 300, 600])

    # All 4 VMs present at each timestamp
    assert np.array_equal(agg["active_vm_count"].values, [4, 4, 4])

    # avg_cpu values are 5, 10, 15, 20 -> mean is 12.5
    assert np.allclose(agg["aggregate_cpu_utilization"].values, [12.5, 12.5, 12.5])


def test_build_aggregate_targets_bounds():
    # Test boundary clipping for extreme values
    records = [
        {"timestamp": 0, "vm_id": "vm_1", "avg_cpu": -10.0},
        {"timestamp": 0, "vm_id": "vm_2", "avg_cpu": 150.0},
    ]
    df = pd.DataFrame(records)
    agg = build_aggregate_targets(df)
    # Mean of -10 and 150 is 70.0 (clipped inside [0, 100])
    assert 0.0 <= agg["aggregate_cpu_utilization"].iloc[0] <= 100.0


def test_check_timestamp_regularity():
    regular_ts = [0, 300, 600, 900]
    res_reg = check_timestamp_regularity(regular_ts, expected_step=300)
    assert res_reg["is_regular"] is True
    assert res_reg["gaps_count"] == 0

    irregular_ts = [0, 300, 900]  # Gap at 600
    res_irreg = check_timestamp_regularity(irregular_ts, expected_step=300)
    assert res_irreg["is_regular"] is False
    assert res_irreg["gaps_count"] == 1


def test_split_train_holdout_chronological_integrity():
    timestamps = [i * 300 for i in range(134)]
    targets = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": [1000] * 134,
        "aggregate_cpu_utilization": [10.0 + (i % 5) for i in range(134)],
    })

    train, holdout = split_train_holdout(targets, train_size=110, holdout_size=24)

    assert len(train) == 110
    assert len(holdout) == 24
    assert train["timestamp"].iloc[-1] < holdout["timestamp"].iloc[0]
    assert train["timestamp"].iloc[-1] == 109 * 300
    assert holdout["timestamp"].iloc[0] == 110 * 300
    assert holdout["timestamp"].iloc[-1] == 133 * 300


def test_split_train_holdout_small_smoke_dataset():
    # Smoke dataset with only 30 timestamps
    targets = pd.DataFrame({
        "timestamp": [i * 300 for i in range(30)],
        "active_vm_count": [50] * 30,
        "aggregate_cpu_utilization": [15.0] * 30,
    })

    train, holdout = split_train_holdout(targets, train_size=110, holdout_size=10)
    assert len(train) == 20
    assert len(holdout) == 10
    assert train["timestamp"].iloc[-1] < holdout["timestamp"].iloc[0]
