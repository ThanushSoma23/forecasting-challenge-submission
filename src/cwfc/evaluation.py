"""Time-aware evaluation and validation utilities for cloud workload forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.forecasting import (
    mean_reverting_forecast,
    moving_average_forecast,
    persistence_forecast,
)
from cwfc.metrics import mae, rmse, smape


def evaluate_temporal_cv(
    train_df: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    horizon: int = 24,
    min_train_size: int = 50,
    step: int = 12,
) -> dict[str, dict[str, Any]]:
    """Evaluate models using expanding-window cross validation on the training set.

    This ensures no future target observations leak into model evaluation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training observations (first 110 observations).
    target_col : str
        Target column to evaluate.
    horizon : int
        Forecast horizon per fold (default: 24 observations = 2 hours).
    min_train_size : int
        Minimum training size for initial fold (default: 50 observations).
    step : int
        Number of steps to advance the forecast origin per fold (default: 12).

    Returns
    -------
    dict[str, dict[str, Any]]
        Validation metrics (MAE, RMSE, sMAPE, relative MAE) for each model.
    """
    n = len(train_df)
    origins = list(range(min_train_size, n - horizon + 1, step))
    if not origins or (n - horizon not in origins):
        origins.append(n - horizon)

    models = {
        "persistence": lambda tr, h: persistence_forecast(tr, h, target_col),
        "moving_average_3": lambda tr, h: moving_average_forecast(tr, h, target_col, window=3),
        "mean_reverting_ar1": lambda tr, h: mean_reverting_forecast(tr, h, target_col, phi=0.90),
    }

    records: dict[str, dict[str, list[float]]] = {
        name: {"mae": [], "rmse": [], "smape": []} for name in models
    }
    horizon_errors: dict[str, list[list[float]]] = {
        name: [[] for _ in range(horizon)] for name in models
    }

    for origin in origins:
        sub_train = train_df.iloc[:origin]
        sub_val = train_df.iloc[origin : origin + horizon]
        y_true = sub_val[target_col].astype(float).values

        for name, forecaster in models.items():
            pred = forecaster(sub_train, horizon).values

            records[name]["mae"].append(mae(y_true, pred))
            records[name]["rmse"].append(rmse(y_true, pred))
            records[name]["smape"].append(smape(y_true, pred))

            for h_idx in range(horizon):
                horizon_errors[name][h_idx].append(abs(y_true[h_idx] - pred[h_idx]))

    # Compute summary metrics relative to persistence
    base_mae = float(np.mean(records["persistence"]["mae"]))
    summary: dict[str, dict[str, Any]] = {}

    for name in models:
        mean_mae = float(np.mean(records[name]["mae"]))
        summary[name] = {
            "mae_mean": mean_mae,
            "mae_std": float(np.std(records[name]["mae"])),
            "rmse_mean": float(np.mean(records[name]["rmse"])),
            "smape_mean": float(np.mean(records[name]["smape"])),
            "rel_mae": mean_mae / base_mae if base_mae > 0 else 1.0,
            "num_folds": len(origins),
            "horizon_mae": [float(np.mean(errs)) for errs in horizon_errors[name]],
        }

    return summary


def evaluate_peak_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 75.0,
) -> dict[str, float]:
    """Measure forecasting error during high-load / peak periods.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values.
    y_pred : np.ndarray
        Forecasted values.
    percentile : float
        Percentile cutoff defining peak periods (default: 75th percentile).

    Returns
    -------
    dict[str, float]
        Overall MAE vs peak MAE and ratio.
    """
    threshold = float(np.percentile(y_true, percentile))
    peak_mask = y_true >= threshold

    overall_mae = mae(y_true, y_pred)
    peak_mae = mae(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else overall_mae
    peak_rmse = rmse(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else rmse(y_true, y_pred)

    return {
        "threshold": threshold,
        "peak_count": int(np.sum(peak_mask)),
        "overall_mae": overall_mae,
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
        "peak_ratio": peak_mae / overall_mae if overall_mae > 0 else 1.0,
    }


def plot_retrospective_evaluation(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: str | Path = "reports/figures/holdout_forecast_vs_actual.png",
) -> str:
    """Generate retrospective forecast vs actual chart for the final holdout.

    Note: This is used only for retrospective reporting after predictions are frozen.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t_train = train_df["timestamp"] / 3600.0
    t_holdout = holdout_df["timestamp"] / 3600.0
    split_t = t_holdout.iloc[0]

    # Target 1: Active VMs
    ax1.plot(t_train, train_df["active_vm_count"], label="Train History", color="#1f77b4")
    ax1.plot(t_holdout, holdout_df["active_vm_count"], label="Actual Holdout", color="#2ca02c")
    ax1.plot(t_holdout, predictions_df["active_vm_count"], label="Forecast (Persistence)", color="#d62728", linestyle="--")
    ax1.axvline(x=split_t, color="black", linestyle=":", label="Forecast Origin (T=110)")
    ax1.set_ylabel("Active VM Count")
    ax1.set_title("Target 1: Active VM Count (Holdout Horizon)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Target 2: Aggregate CPU
    ax2.plot(t_train, train_df["aggregate_cpu_utilization"], label="Train History", color="#1f77b4")
    ax2.plot(t_holdout, holdout_df["aggregate_cpu_utilization"], label="Actual Holdout", color="#2ca02c")
    ax2.plot(t_holdout, predictions_df["aggregate_cpu_utilization"], label="Forecast (Mean-Reversion)", color="#d62728", linestyle="--")
    ax2.axvline(x=split_t, color="black", linestyle=":", label="Forecast Origin (T=110)")
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Aggregate CPU (%)")
    ax2.set_title("Target 2: Aggregate CPU Utilization (%) (Holdout Horizon)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out), dpi=200)
    plt.close(fig)
    return str(out)
