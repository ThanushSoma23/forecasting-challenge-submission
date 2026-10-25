"""Run exploratory analysis, temporal validation benchmark, and figure generation."""

from pathlib import Path
import numpy as np
import pandas as pd

from cwfc.data import (
    build_aggregate_targets,
    load_cpu_readings,
    load_vm_metadata,
    split_train_holdout,
)
from cwfc.analysis import (
    analyze_workload_segments,
    compute_workload_summary,
    generate_eda_figures,
)
from cwfc.evaluation import (
    evaluate_peak_performance,
    evaluate_temporal_cv,
    plot_retrospective_evaluation,
)
from cwfc.metrics import mae, rmse, smape


def main():
    print("Loading official dataset...")
    readings = load_cpu_readings("challenge_data/vm_cpu_readings.csv.gz")
    metadata = load_vm_metadata("challenge_data/vm_metadata.csv.gz")
    targets = build_aggregate_targets(readings)

    train_df, holdout_df = split_train_holdout(targets, train_size=110, holdout_size=24)

    # 1. Generate EDA Figures
    print("\nGenerating EDA figures in reports/figures/...")
    fig_paths = generate_eda_figures(readings, metadata, targets, output_dir="reports/figures")
    for p in fig_paths:
        print(f"  Saved: {p}")

    # 2. Expanding-window temporal cross validation (training window only)
    print("\nRunning expanding-window cross validation on the first 110 observations...")
    val_summary = evaluate_temporal_cv(
        train_df,
        target_col="aggregate_cpu_utilization",
        horizon=24,
        min_train_size=50,
        step=12,
    )

    print("\n=== EXPANDING-WINDOW CROSS-VALIDATION (110 Train Observations) ===")
    for model_name, stats in val_summary.items():
        print(
            f"{model_name:20s} | MAE: {stats['mae_mean']:.4f} (+/- {stats['mae_std']:.4f}) | "
            f"RMSE: {stats['rmse_mean']:.4f} | sMAPE: {stats['smape_mean']:.3f}% | Rel MAE: {stats['rel_mae']:.4f}"
        )

    # 3. Retrospective Holdout Evaluation (Final reporting only)
    predictions_df = pd.read_csv("predictions.csv")
    retro_mae_cpu = mae(
        holdout_df["aggregate_cpu_utilization"].values,
        predictions_df["aggregate_cpu_utilization"].values,
    )
    retro_rmse_cpu = rmse(
        holdout_df["aggregate_cpu_utilization"].values,
        predictions_df["aggregate_cpu_utilization"].values,
    )
    retro_smape_cpu = smape(
        holdout_df["aggregate_cpu_utilization"].values,
        predictions_df["aggregate_cpu_utilization"].values,
    )

    base_pred = np.full(
        len(holdout_df), train_df["aggregate_cpu_utilization"].iloc[-1]
    )
    retro_mae_base = mae(holdout_df["aggregate_cpu_utilization"].values, base_pred)
    retro_rel_mae = retro_mae_cpu / retro_mae_base

    print("\n=== RETROSPECTIVE HOLDOUT EVALUATION (FINAL 24 OBSERVATIONS) ===")
    print(f"Mean-Reverting AR(1) Holdout MAE:  {retro_mae_cpu:.4f}")
    print(f"Mean-Reverting AR(1) Holdout RMSE: {retro_rmse_cpu:.4f}")
    print(f"Mean-Reverting AR(1) Holdout sMAPE: {retro_smape_cpu:.3f}%")
    print(f"Persistence Baseline Holdout MAE:  {retro_mae_base:.4f}")
    print(f"Holdout Relative MAE:              {retro_rel_mae:.4f} ({(1 - retro_rel_mae)*100:.1f}% improvement)")

    # Peak load evaluation
    peak_stats = evaluate_peak_performance(
        holdout_df["aggregate_cpu_utilization"].values,
        predictions_df["aggregate_cpu_utilization"].values,
        percentile=75.0,
    )
    print(
        f"\nPeak Period Evaluation (Holdout > P75={peak_stats['threshold']:.3f}%):\n"
        f"  Overall MAE: {peak_stats['overall_mae']:.4f}\n"
        f"  Peak MAE:    {peak_stats['peak_mae']:.4f}\n"
        f"  Peak Ratio:  {peak_stats['peak_ratio']:.4f}"
    )

    # Retrospective plot
    fig_retro = plot_retrospective_evaluation(
        train_df,
        holdout_df,
        predictions_df,
        output_path="reports/figures/holdout_forecast_vs_actual.png",
    )
    print(f"\nRetrospective plot saved: {fig_retro}")

    # Summary
    summary = compute_workload_summary(readings, metadata)
    print("\n=== WORKLOAD DATASET SUMMARY ===")
    print(f"Total Readings: {summary['total_readings']:,}")
    print(f"Unique VMs:     {summary['unique_vms']:,}")
    print(f"Avg CPU Mean:   {summary['avg_cpu_mean']:.2f}% (Std: {summary['avg_cpu_std']:.2f}%)")
    print(f"Avg CPU Median: {summary['avg_cpu_median']:.2f}%")
    print(f"Idle (<5% CPU): {summary['idle_fraction_lt_5pct']*100:.2f}%")


if __name__ == "__main__":
    main()
