"""Data ingestion, validation, and aggregation utilities for cloud workload telemetry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd


REQUIRED_READINGS_COLUMNS: list[str] = [
    "timestamp",
    "vm_id",
    "min_cpu",
    "max_cpu",
    "avg_cpu",
]

REQUIRED_METADATA_COLUMNS: list[str] = [
    "vm_id",
    "subscription_id",
    "deployment_id",
    "timestamp_vm_created",
    "timestamp_vm_deleted",
    "max_cpu",
    "avg_cpu",
    "p95_max_cpu",
    "vm_category",
    "virtual_core_bucket",
    "memory_gb_bucket",
]


def load_cpu_readings(file_path: str | Path) -> pd.DataFrame:
    """Load and validate VM-level CPU readings telemetry.

    Parameters
    ----------
    file_path : str | Path
        Path to `vm_cpu_readings.csv.gz` or `.csv`.

    Returns
    -------
    pd.DataFrame
        Validated CPU readings DataFrame sorted by timestamp.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Readings file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = set(REQUIRED_READINGS_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Readings file missing required columns: {sorted(missing_cols)}")

    # Ensure numeric columns
    numeric_cols = ["timestamp", "min_cpu", "max_cpu", "avg_cpu"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Readings column {col} contains non-numeric or NaN values")

    # Ensure valid ranges
    if (df["min_cpu"] < 0).any() or (df["max_cpu"] > 100).any():
        raise ValueError("CPU readings exceed standard range [0, 100]")
    if (df["min_cpu"] > df["max_cpu"]).any():
        raise ValueError("Inconsistent readings: min_cpu > max_cpu found")

    return df.sort_values(by=["timestamp", "vm_id"]).reset_index(drop=True)


def load_vm_metadata(file_path: str | Path) -> pd.DataFrame:
    """Load and validate VM metadata.

    Parameters
    ----------
    file_path : str | Path
        Path to `vm_metadata.csv.gz` or `.csv`.

    Returns
    -------
    pd.DataFrame
        Validated VM metadata DataFrame.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = set(REQUIRED_METADATA_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata file missing required columns: {sorted(missing_cols)}")

    return df.reset_index(drop=True)


def build_aggregate_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Construct the challenge forecasting targets from VM-level CPU readings.

    Targets:
    1. `active_vm_count`: count of unique VM IDs with a valid observation at timestamp t.
    2. `aggregate_cpu_utilization`: mean of the VM-level avg_cpu values at timestamp t,
       bounded in [0.0, 100.0].

    Parameters
    ----------
    readings : pd.DataFrame
        VM-level readings DataFrame containing 'timestamp', 'vm_id', and 'avg_cpu'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'active_vm_count', 'aggregate_cpu_utilization']
        sorted chronologically.
    """
    for col in ["timestamp", "vm_id", "avg_cpu"]:
        if col not in readings.columns:
            raise ValueError(f"Missing column '{col}' needed for aggregation")

    if readings.empty:
        raise ValueError("Cannot aggregate empty readings DataFrame")

    # Target 1: active_vm_count (count distinct VM per timestamp)
    # Target 2: aggregate_cpu_utilization (mean avg_cpu per timestamp)
    agg = readings.groupby("timestamp", as_index=False).agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    )

    agg = agg.sort_values(by="timestamp").reset_index(drop=True)

    # Enforce domain constraints
    agg["active_vm_count"] = agg["active_vm_count"].astype(int)
    agg["active_vm_count"] = np.maximum(0, agg["active_vm_count"])
    agg["aggregate_cpu_utilization"] = np.clip(
        agg["aggregate_cpu_utilization"].astype(float), 0.0, 100.0
    )

    return agg


def check_timestamp_regularity(
    timestamps: Sequence[int | float], expected_step: int = 300
) -> dict[str, object]:
    """Inspect timestamp regularity, interval step, and gaps.

    Parameters
    ----------
    timestamps : Sequence[int | float]
        Chronological timestamp sequence.
    expected_step : int
        Expected sampling interval in seconds (default: 300 for 5 minutes).

    Returns
    -------
    dict[str, object]
        Summary containing 'is_regular', 'detected_steps', 'min_step', 'max_step', 'gaps_count'.
    """
    ts = np.asarray(timestamps)
    if len(ts) < 2:
        return {
            "is_regular": True,
            "detected_steps": [],
            "min_step": None,
            "max_step": None,
            "gaps_count": 0,
        }

    diffs = np.diff(ts)
    unique_steps = np.unique(diffs).tolist()
    is_regular = len(unique_steps) == 1 and unique_steps[0] == expected_step
    gaps_count = int(np.sum(diffs > expected_step))

    return {
        "is_regular": is_regular,
        "detected_steps": unique_steps,
        "min_step": int(np.min(diffs)),
        "max_step": int(np.max(diffs)),
        "gaps_count": gaps_count,
    }


def split_train_holdout(
    targets: pd.DataFrame,
    train_size: int = 110,
    holdout_size: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aggregate targets chronologically into public train and holdout sets.

    Parameters
    ----------
    targets : pd.DataFrame
        Aggregated time series DataFrame.
    train_size : int
        Number of observations in the training window (default: 110).
    holdout_size : int
        Number of observations in the holdout window (default: 24).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, holdout_df) strictly split by chronological index.
    """
    if len(targets) < train_size + holdout_size:
        # Support smaller replacement / smoke test datasets gracefully
        if len(targets) <= holdout_size:
            raise ValueError(
                f"Targets length ({len(targets)}) must be greater than holdout_size ({holdout_size})"
            )
        actual_train_size = len(targets) - holdout_size
        train = targets.iloc[:actual_train_size].copy().reset_index(drop=True)
        holdout = targets.iloc[actual_train_size:].copy().reset_index(drop=True)
        return train, holdout

    train = targets.iloc[:train_size].copy().reset_index(drop=True)
    holdout = targets.iloc[train_size : train_size + holdout_size].copy().reset_index(drop=True)

    # Verify chronological integrity
    if not train["timestamp"].is_monotonic_increasing:
        raise ValueError("Train timestamps are not monotonically increasing")
    if not holdout["timestamp"].is_monotonic_increasing:
        raise ValueError("Holdout timestamps are not monotonically increasing")
    if train["timestamp"].iloc[-1] >= holdout["timestamp"].iloc[0]:
        raise ValueError("Data leakage detected: train timestamp overlaps with holdout timestamp")

    return train, holdout


def verify_sha256(file_path: str | Path, expected_hash: str) -> bool:
    """Compute and verify the SHA256 checksum of a file.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to verify.
    expected_hash : str
        Hex string of expected SHA256 hash.

    Returns
    -------
    bool
        True if the file's SHA256 matches expected_hash, False otherwise.
    """
    path = Path(file_path)
    if not path.is_file():
        return False
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_hash.strip().lower()


def verify_dataset_checksums(data_dir: str | Path) -> dict[str, bool]:
    """Verify all checksums listed in SHA256SUMS within data_dir.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing `SHA256SUMS` and dataset files.

    Returns
    -------
    dict[str, bool]
        Mapping of filename to verification status.
    """
    dir_path = Path(data_dir)
    sums_file = dir_path / "SHA256SUMS"
    if not sums_file.is_file():
        raise FileNotFoundError(f"SHA256SUMS not found in {dir_path}")

    results = {}
    with open(sums_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_hash, fname = parts[0], parts[1].strip()
                file_target = dir_path / fname
                results[fname] = verify_sha256(file_target, expected_hash)
    return results
