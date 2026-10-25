"""Exploratory workload analysis, profiling, and visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_workload_summary(
    readings: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute comprehensive summary metrics for cloud VM telemetry.

    Parameters
    ----------
    readings : pd.DataFrame
        VM-level CPU readings telemetry.
    metadata : pd.DataFrame | None
        VM metadata (optional).

    Returns
    -------
    dict[str, Any]
        Dictionary of workload summary statistics.
    """
    total_readings = len(readings)
    unique_vms = readings["vm_id"].nunique()
    unique_timestamps = np.sort(readings["timestamp"].unique())
    num_timestamps = len(unique_timestamps)
    time_span_seconds = int(unique_timestamps[-1] - unique_timestamps[0]) if num_timestamps > 1 else 0
    time_span_hours = time_span_seconds / 3600.0

    steps = np.diff(unique_timestamps) if num_timestamps > 1 else np.array([0])
    cadence_seconds = int(steps[0]) if len(steps) > 0 and len(np.unique(steps)) == 1 else None

    # CPU percentiles across all VM readings
    avg_cpu_quantiles = readings["avg_cpu"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    max_cpu_quantiles = readings["max_cpu"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

    idle_fraction = float((readings["avg_cpu"] < 5.0).mean())
    high_util_fraction = float((readings["avg_cpu"] > 50.0).mean())
    extreme_burst_fraction = float((readings["max_cpu"] > 90.0).mean())

    summary: dict[str, Any] = {
        "total_readings": total_readings,
        "unique_vms": unique_vms,
        "num_timestamps": num_timestamps,
        "time_span_seconds": time_span_seconds,
        "time_span_hours": round(time_span_hours, 2),
        "cadence_seconds": cadence_seconds,
        "avg_cpu_mean": float(readings["avg_cpu"].mean()),
        "avg_cpu_std": float(readings["avg_cpu"].std()),
        "avg_cpu_median": float(readings["avg_cpu"].median()),
        "avg_cpu_quantiles": avg_cpu_quantiles,
        "max_cpu_quantiles": max_cpu_quantiles,
        "idle_fraction_lt_5pct": idle_fraction,
        "high_util_fraction_gt_50pct": high_util_fraction,
        "extreme_burst_fraction_gt_90pct": extreme_burst_fraction,
    }

    if metadata is not None:
        summary["metadata_vms"] = len(metadata)
        summary["categories"] = metadata["vm_category"].value_counts(normalize=True).to_dict()
        summary["core_buckets"] = metadata["virtual_core_bucket"].value_counts(normalize=True).to_dict()
        summary["memory_buckets"] = metadata["memory_gb_bucket"].value_counts(normalize=True).to_dict()
        summary["unique_subscriptions"] = int(metadata["subscription_id"].nunique())
        summary["unique_deployments"] = int(metadata["deployment_id"].nunique())

    return summary


def analyze_workload_segments(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze CPU utilization and workload profiles across VM categories and sizing tiers.

    Parameters
    ----------
    readings : pd.DataFrame
        VM-level CPU readings.
    metadata : pd.DataFrame
        VM metadata.

    Returns
    -------
    pd.DataFrame
        Segment-level aggregated statistics.
    """
    merged = readings.merge(metadata[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]], on="vm_id")

    grouped = merged.groupby("vm_category").agg(
        vm_count=("vm_id", "nunique"),
        avg_cpu_mean=("avg_cpu", "mean"),
        avg_cpu_std=("avg_cpu", "std"),
        avg_cpu_p50=("avg_cpu", "median"),
        avg_cpu_p95=("avg_cpu", lambda s: np.percentile(s, 95)),
        max_cpu_mean=("max_cpu", "mean"),
        max_cpu_p95=("max_cpu", lambda s: np.percentile(s, 95)),
    ).reset_index()

    return grouped


def analyze_autocorrelation(series: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    """Compute autocorrelation for lags 1 through max_lag.

    Parameters
    ----------
    series : pd.Series
        Univariate target series.
    max_lag : int
        Maximum lag to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with lag, lag_minutes, and autocorrelation value.
    """
    lags = list(range(1, max_lag + 1))
    autocorrs = [series.autocorr(lag=k) for k in lags]
    return pd.DataFrame({
        "lag": lags,
        "lag_minutes": [k * 5 for k in lags],
        "autocorrelation": autocorrs,
    })


def generate_eda_figures(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
) -> list[str]:
    """Generate and save publication-quality EDA figures for reports and notebooks.

    Parameters
    ----------
    readings : pd.DataFrame
        VM-level readings.
    metadata : pd.DataFrame
        VM metadata.
    targets : pd.DataFrame
        Aggregated time series targets.
    output_dir : str | Path
        Directory where figures will be saved.

    Returns
    -------
    list[str]
        List of generated figure file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    # Style configuration
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    })

    # Figure 1: Target time series (Active VMs & Aggregate CPU Utilization)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    timestamps_hours = targets["timestamp"] / 3600.0

    ax1.plot(timestamps_hours, targets["active_vm_count"], color="#1f77b4", lw=2)
    ax1.axvline(x=110 * 300 / 3600.0, color="#d62728", linestyle="--", label="Train / Holdout Origin (T=110)")
    ax1.set_ylabel("Active VM Count")
    ax1.set_title("Target 1: Active VM Count Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.plot(timestamps_hours, targets["aggregate_cpu_utilization"], color="#2ca02c", lw=2, label="Mean CPU (%)")
    ax2.axvline(x=110 * 300 / 3600.0, color="#d62728", linestyle="--", label="Train / Holdout Origin (T=110)")
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Aggregate CPU (%)")
    ax2.set_title("Target 2: Aggregate Mean CPU Utilization Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    f1_path = str(out_path / "target_series.png")
    fig.savefig(f1_path, dpi=200)
    plt.close(fig)
    generated_files.append(f1_path)

    # Figure 2: VM-level CPU Distribution (Right skew, idle dominance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    sample_readings = readings["avg_cpu"].sample(n=min(50000, len(readings)), random_state=42)
    ax1.hist(sample_readings, bins=50, color="#4a7c59", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Average CPU Utilization (%)")
    ax1.set_ylabel("Reading Frequency (Sample of 50k)")
    ax1.set_title("Histogram: VM-Level Average CPU Utilization")
    ax1.axvline(sample_readings.median(), color="red", linestyle="--", label=f"Median ({sample_readings.median():.2f}%)")
    ax1.axvline(sample_readings.mean(), color="orange", linestyle="-", label=f"Mean ({sample_readings.mean():.2f}%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Boxplot by VM category
    merged = readings.sample(n=min(50000, len(readings)), random_state=42).merge(
        metadata[["vm_id", "vm_category"]], on="vm_id"
    )
    categories = [cat for cat in ["Interactive", "Delay-insensitive", "Unknown"] if cat in merged["vm_category"].unique()]
    box_data = [merged[merged["vm_category"] == c]["avg_cpu"].dropna().values for c in categories]
    ax2.boxplot(box_data, tick_labels=categories, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#a1c9f4", color="#1f77b4"))
    ax2.set_ylabel("Average CPU Utilization (%)")
    ax2.set_title("CPU Utilization by Workload Category (No Fliers)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    f2_path = str(out_path / "cpu_distribution.png")
    fig.savefig(f2_path, dpi=200)
    plt.close(fig)
    generated_files.append(f2_path)

    # Figure 3: Autocorrelation & Seasonality of Aggregate CPU Utilization
    fig, ax = plt.subplots(figsize=(8, 4))
    train_cpu = targets.iloc[:110]["aggregate_cpu_utilization"]
    autocorr_df = analyze_autocorrelation(train_cpu, max_lag=24)
    ax.stem(autocorr_df["lag_minutes"], autocorr_df["autocorrelation"], basefmt=" ")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag Horizon (Minutes)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function (ACF) of Aggregate CPU (Training Window)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    f3_path = str(out_path / "autocorrelation.png")
    fig.savefig(f3_path, dpi=200)
    plt.close(fig)
    generated_files.append(f3_path)

    return generated_files
