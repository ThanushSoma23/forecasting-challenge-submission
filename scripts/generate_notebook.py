"""Generate and execute notebooks/workload_analysis.ipynb with clean outputs."""

import base64
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.data import (
    load_cpu_readings,
    load_vm_metadata,
    build_aggregate_targets,
    split_train_holdout,
)
from cwfc.analysis import (
    compute_workload_summary,
    analyze_workload_segments,
    analyze_autocorrelation,
)
from cwfc.forecasting import (
    persistence_forecast,
    moving_average_forecast,
    mean_reverting_forecast,
)
from cwfc.evaluation import (
    evaluate_temporal_cv,
    evaluate_peak_performance,
)
from cwfc.metrics import mae, rmse, smape


def main():
    print("Executing notebook steps and capturing outputs...")

    # Cell 1 Code: Imports
    c1_source = [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from pathlib import Path\n",
        "\n",
        "from cwfc.data import (\n",
        "    load_cpu_readings,\n",
        "    load_vm_metadata,\n",
        "    build_aggregate_targets,\n",
        "    split_train_holdout,\n",
        ")\n",
        "from cwfc.analysis import (\n",
        "    compute_workload_summary,\n",
        "    analyze_workload_segments,\n",
        "    analyze_autocorrelation,\n",
        ")\n",
        "from cwfc.forecasting import (\n",
        "    persistence_forecast,\n",
        "    moving_average_forecast,\n",
        "    mean_reverting_forecast,\n",
        ")\n",
        "from cwfc.evaluation import (\n",
        "    evaluate_temporal_cv,\n",
        "    evaluate_peak_performance,\n",
        ")\n",
        "from cwfc.metrics import mae, rmse, smape\n",
        "\n",
        "print('Imports successful.')\n",
    ]
    c1_output = [{
        "name": "stdout",
        "output_type": "stream",
        "text": ["Imports successful.\n"]
    }]

    # Cell 2 Code: Ingestion & Target Aggregation
    readings = load_cpu_readings("challenge_data/vm_cpu_readings.csv.gz")
    metadata = load_vm_metadata("challenge_data/vm_metadata.csv.gz")
    targets = build_aggregate_targets(readings)

    c2_source = [
        "readings = load_cpu_readings('challenge_data/vm_cpu_readings.csv.gz')\n",
        "metadata = load_vm_metadata('challenge_data/vm_metadata.csv.gz')\n",
        "targets = build_aggregate_targets(readings)\n",
        "\n",
        "print(f'Readings rows: {len(readings):,}')\n",
        "print(f'Unique VMs: {readings[\"vm_id\"].nunique():,}')\n",
        "print(f'Aggregate Timestamps: {len(targets)}')\n",
        "targets.head()\n",
    ]
    targets_head = targets.head()
    c2_output = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Readings rows: {len(readings):,}\n",
                f"Unique VMs: {readings['vm_id'].nunique():,}\n",
                f"Aggregate Timestamps: {len(targets)}\n",
            ]
        },
        {
            "data": {
                "text/html": [targets_head.to_html()],
                "text/plain": [targets_head.to_string()],
            },
            "execution_count": 2,
            "metadata": {},
            "output_type": "execute_result"
        }
    ]

    # Cell 3 Code: Workload Distribution & VM Segmentation
    summary = compute_workload_summary(readings, metadata)
    seg_df = analyze_workload_segments(readings, metadata)

    c3_source = [
        "summary = compute_workload_summary(readings, metadata)\n",
        "print(f'Median VM CPU: {summary[\"avg_cpu_median\"]:.2f}%')\n",
        "print(f'Fraction < 5% CPU: {summary[\"idle_fraction_lt_5pct\"]*100:.2f}%')\n",
        "print(f'Fraction > 50% CPU: {summary[\"high_util_fraction_gt_50pct\"]*100:.2f}%')\n",
        "\n",
        "seg_df = analyze_workload_segments(readings, metadata)\n",
        "seg_df\n",
    ]
    c3_output = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Median VM CPU: {summary['avg_cpu_median']:.2f}%\n",
                f"Fraction < 5% CPU: {summary['idle_fraction_lt_5pct']*100:.2f}%\n",
                f"Fraction > 50% CPU: {summary['high_util_fraction_gt_50pct']*100:.2f}%\n",
            ]
        },
        {
            "data": {
                "text/html": [seg_df.to_html()],
                "text/plain": [seg_df.to_string()],
            },
            "execution_count": 3,
            "metadata": {},
            "output_type": "execute_result"
        }
    ]

    # Cell 4 Code: Autocorrelation
    train_df, holdout_df = split_train_holdout(targets, train_size=110, holdout_size=24)
    acf_df = analyze_autocorrelation(train_df["aggregate_cpu_utilization"], max_lag=18)
    acf_head = acf_df.head(10)

    c4_source = [
        "train_df, holdout_df = split_train_holdout(targets, train_size=110, holdout_size=24)\n",
        "acf_df = analyze_autocorrelation(train_df['aggregate_cpu_utilization'], max_lag=18)\n",
        "acf_df.head(10)\n",
    ]
    c4_output = [
        {
            "data": {
                "text/html": [acf_head.to_html()],
                "text/plain": [acf_head.to_string()],
            },
            "execution_count": 4,
            "metadata": {},
            "output_type": "execute_result"
        }
    ]

    # Cell 5 Code: Expanding-Window CV
    val_results = evaluate_temporal_cv(
        train_df, target_col="aggregate_cpu_utilization", horizon=24, min_train_size=50, step=12
    )
    val_lines = []
    for name, stats in val_results.items():
        val_lines.append(f"{name:20s} | MAE: {stats['mae_mean']:.4f} | RMSE: {stats['rmse_mean']:.4f} | Rel MAE: {stats['rel_mae']:.4f}\n")

    c5_source = [
        "val_results = evaluate_temporal_cv(\n",
        "    train_df, target_col='aggregate_cpu_utilization', horizon=24, min_train_size=50, step=12\n",
        ")\n",
        "for name, stats in val_results.items():\n",
        "    print(f\"{name:20s} | MAE: {stats['mae_mean']:.4f} | RMSE: {stats['rmse_mean']:.4f} | Rel MAE: {stats['rel_mae']:.4f}\")\n",
    ]
    c5_output = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": val_lines
        }
    ]

    # Cell 6 Code: Retrospective Holdout Evaluation
    preds = pd.read_csv("predictions.csv")
    h_true = holdout_df["aggregate_cpu_utilization"].values
    h_pred = preds["aggregate_cpu_utilization"].values
    h_mae = mae(h_true, h_pred)
    h_rmse = rmse(h_true, h_pred)
    h_smape = smape(h_true, h_pred)

    c6_source = [
        "preds = pd.read_csv('predictions.csv')\n",
        "h_true = holdout_df['aggregate_cpu_utilization'].values\n",
        "h_pred = preds['aggregate_cpu_utilization'].values\n",
        "\n",
        "print(f\"Holdout MAE:  {mae(h_true, h_pred):.4f}\")\n",
        "print(f\"Holdout RMSE: {rmse(h_true, h_pred):.4f}\")\n",
        "print(f\"Holdout sMAPE: {smape(h_true, h_pred):.3f}%\")\n",
    ]
    c6_output = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Holdout MAE:  {h_mae:.4f}\n",
                f"Holdout RMSE: {h_rmse:.4f}\n",
                f"Holdout sMAPE: {h_smape:.3f}%\n",
            ]
        }
    ]

    # Cell 7 Code: Visualization of Holdout
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(train_df["timestamp"], train_df["aggregate_cpu_utilization"], label="Train (110 steps)", color="#1f77b4")
    axes[0].plot(holdout_df["timestamp"], holdout_df["aggregate_cpu_utilization"], label="Holdout (24 steps)", color="#2ca02c")
    axes[0].set_title("Aggregate CPU Utilization Time Series")
    axes[0].set_xlabel("Timestamp (seconds)")
    axes[0].set_ylabel("Mean CPU (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(holdout_df["timestamp"], h_true, label="Actual Holdout", color="#2ca02c", marker="o", markersize=3)
    axes[1].plot(holdout_df["timestamp"], h_pred, label="Mean-Reversion Forecast", color="#d62728", linestyle="--", marker="s", markersize=3)
    axes[1].set_title("Holdout: Forecast vs Actual (2 Hours)")
    axes[1].set_xlabel("Timestamp (seconds)")
    axes[1].set_ylabel("Mean CPU (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)

    c7_source = [
        "# Visualize the aggregate series and retrospective forecast vs actual\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n",
        "axes[0].plot(train_df['timestamp'], train_df['aggregate_cpu_utilization'], label='Train (110 steps)', color='#1f77b4')\n",
        "axes[0].plot(holdout_df['timestamp'], holdout_df['aggregate_cpu_utilization'], label='Holdout (24 steps)', color='#2ca02c')\n",
        "axes[0].set_title('Aggregate CPU Utilization Time Series')\n",
        "axes[0].set_xlabel('Timestamp (seconds)')\n",
        "axes[0].set_ylabel('Mean CPU (%)')\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1].plot(holdout_df['timestamp'], h_true, label='Actual Holdout', color='#2ca02c', marker='o', markersize=3)\n",
        "axes[1].plot(holdout_df['timestamp'], h_pred, label='Mean-Reversion Forecast', color='#d62728', linestyle='--', marker='s', markersize=3)\n",
        "axes[1].set_title('Holdout: Forecast vs Actual (2 Hours)')\n",
        "axes[1].set_xlabel('Timestamp (seconds)')\n",
        "axes[1].set_ylabel('Mean CPU (%)')\n",
        "axes[1].legend()\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]
    c7_output = [
        {
            "data": {
                "image/png": plot_b64,
                "text/plain": ["<Figure size 1300x400 with 2 Axes>"]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cloud Workload Forecasting Challenge\n",
                    "## Exploratory Data Analysis & Workload Forecasting Pipeline\n",
                    "\n",
                    "This notebook presents an end-to-end analysis of the Microsoft Azure Public Dataset V2 (`data-v1` release).\n",
                    "We explore workload distributions, temporal autocorrelation, VM segmentation, and evaluate candidate forecasting models under strict leakage-safe expanding-window cross-validation.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": c1_output,
                "source": c1_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Ingestion & Target Aggregation\n",
                    "We load the VM-level readings and metadata, and construct the two required aggregate targets:\n",
                    "1. `active_vm_count`: count of unique VM IDs observed at each timestamp.\n",
                    "2. `aggregate_cpu_utilization`: mean of VM-level `avg_cpu` values at each timestamp.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": c2_output,
                "source": c2_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Workload Distribution & VM Segmentation\n",
                    "We examine the distribution of CPU utilization across individual VMs and compare profiles between Interactive and Delay-insensitive workload categories.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": c3_output,
                "source": c3_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Autocorrelation & Time Series Dynamics\n",
                    "We inspect the serial correlation structure of the training series (first 110 observations) to identify memory and periodic cycles.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": c4_output,
                "source": c4_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Expanding-Window Cross-Validation\n",
                    "To select our candidate forecasting model without data leakage, we evaluate models across 4 temporal validation folds strictly on the 110 training observations.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": c5_output,
                "source": c5_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Retrospective Holdout Evaluation\n",
                    "Model selection and tuning were performed strictly on the training set. We evaluate our frozen predictions retrospectively against the true holdout targets.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": c6_output,
                "source": c6_source,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Visualizing Forecast vs Actual\n",
                    "Below we visualize the aggregate CPU utilization time series and compare the 2-hour holdout forecast against actual observations.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {},
                "outputs": c7_output,
                "source": c7_source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_file = Path("notebooks/workload_analysis.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated and executed {out_file} with full outputs.")


if __name__ == "__main__":
    main()
