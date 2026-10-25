"""Prediction CLI pipeline for generating official challenge forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from cwfc.data import (
    build_aggregate_targets,
    load_cpu_readings,
    split_train_holdout,
)
from cwfc.forecasting import mean_reverting_forecast, persistence_forecast


REQUIRED_PREDICTION_COLUMNS: list[str] = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
]


def validate_predictions_df(
    preds_df: pd.DataFrame,
    expected_timestamps: Sequence[int | float] | None = None,
    expected_rows: int = 24,
) -> None:
    """Validate prediction DataFrame according to the submission contract.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame to validate.
    expected_timestamps : Sequence[int | float] | None
        Expected chronological timestamps, if known.
    expected_rows : int
        Expected number of prediction rows (default: 24).
    """
    if list(preds_df.columns) != REQUIRED_PREDICTION_COLUMNS:
        raise ValueError(
            f"Prediction columns must be exactly {REQUIRED_PREDICTION_COLUMNS}, got {list(preds_df.columns)}"
        )

    if len(preds_df) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} prediction rows, got {len(preds_df)}")

    # Check finite numeric values
    for col in REQUIRED_PREDICTION_COLUMNS:
        if not np.issubdtype(preds_df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must contain numeric values.")
        if not np.all(np.isfinite(preds_df[col])):
            raise ValueError(f"Column '{col}' contains non-finite values (NaN or Inf).")

    # Check timestamps
    ts = preds_df["timestamp"]
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly monotonic increasing.")
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps found in predictions.")

    if expected_timestamps is not None:
        if not np.array_equal(ts.values, np.asarray(expected_timestamps)):
            raise ValueError("Timestamps do not match expected holdout timestamps.")

    # Domain range constraints
    if (preds_df["active_vm_count"] < 0).any():
        raise ValueError("Constraint violation: active_vm_count must be >= 0.")

    cpu = preds_df["aggregate_cpu_utilization"]
    if (cpu < 0.0).any() or (cpu > 100.0).any():
        raise ValueError("Constraint violation: aggregate_cpu_utilization must be in [0, 100].")


def validate_predictions_file(
    file_path: str | Path,
    expected_timestamps: Sequence[int | float] | None = None,
    expected_rows: int = 24,
) -> pd.DataFrame:
    """Load and validate a predictions CSV file from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)
    validate_predictions_df(df, expected_timestamps=expected_timestamps, expected_rows=expected_rows)
    return df


def generate_forecasts(
    readings_path: str | Path,
    train_size: int = 110,
    holdout_size: int = 24,
) -> pd.DataFrame:
    """Run end-to-end forecasting pipeline to generate holdout predictions.

    Parameters
    ----------
    readings_path : str | Path
        Path to data directory or vm_cpu_readings.csv.gz.
    train_size : int
        Number of training observations (default: 110).
    holdout_size : int
        Number of holdout observations to forecast (default: 24).

    Returns
    -------
    pd.DataFrame
        Validated predictions DataFrame.
    """
    path = Path(readings_path)
    if path.is_dir():
        file_to_load = path / "vm_cpu_readings.csv.gz"
        if not file_to_load.is_file():
            file_to_load = path / "vm_cpu_readings.csv"
    else:
        file_to_load = path

    readings = load_cpu_readings(file_to_load)
    targets = build_aggregate_targets(readings)
    train_df, holdout_template = split_train_holdout(targets, train_size=train_size, holdout_size=holdout_size)

    horizon = len(holdout_template)

    # 1. Target 1: active_vm_count (persistence baseline)
    vm_preds = persistence_forecast(train_df, horizon, target_col="active_vm_count")

    # 2. Target 2: aggregate_cpu_utilization (mean-reverting AR(1))
    cpu_preds = mean_reverting_forecast(train_df, horizon, target_col="aggregate_cpu_utilization", phi=0.90)

    out_df = pd.DataFrame({
        "timestamp": holdout_template["timestamp"].values,
        "active_vm_count": np.round(vm_preds.values).astype(int),
        "aggregate_cpu_utilization": cpu_preds.values,
    })

    validate_predictions_df(out_df, expected_timestamps=holdout_template["timestamp"].values, expected_rows=horizon)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cloud workload forecasts for take-home challenge.")
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to challenge data directory or vm_cpu_readings.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to write predictions.csv",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    preds = generate_forecasts(args.data)

    out_file = Path(args.output)
    try:
        preds.to_csv(out_file, index=False)
        print(f"Predictions written to: {out_file.resolve()}")
    except PermissionError:
        print(
            f"Error: Could not write to '{out_file.name}' because the file is locked by another program "
            "(e.g., Microsoft Excel). Please close the file in Excel and run again."
        )
        raise

    validate_predictions_file(out_file, expected_timestamps=preds["timestamp"].values)
    print("Submission schema verification: PASSED (24 rows, valid schema).")


if __name__ == "__main__":
    main()
