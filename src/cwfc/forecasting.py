"""Time-series forecasting models and baselines for cloud workload telemetry.

This module implements:
1. Persistence / Last-Value Baseline: projects the most recent observation forward.
2. Moving Average Baseline: projects the rolling mean of the last K observations.
3. Mean-Reversion Forecasting Rule: combines the last observed value with gradual
   decay toward the historical training mean using a fixed decay factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(
    train: pd.DataFrame,
    horizon: int,
    target_col: str,
) -> pd.Series:
    """Forecast by repeating the last observed value across the entire horizon.

    Parameters
    ----------
    train : pd.DataFrame
        Training data available at the forecast origin.
    horizon : int
        Number of future steps to forecast.
    target_col : str
        Target column name.

    Returns
    -------
    pd.Series
        Predicted values for the forecast horizon.
    """
    if train.empty:
        raise ValueError("Training data cannot be empty.")
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    last_val = float(train[target_col].iloc[-1])
    preds = np.full(horizon, last_val, dtype=float)
    return enforce_constraints(preds, target_col)


def moving_average_forecast(
    train: pd.DataFrame,
    horizon: int,
    target_col: str,
    window: int = 3,
) -> pd.Series:
    """Forecast by projecting the mean of the most recent K observations.

    Parameters
    ----------
    train : pd.DataFrame
        Training data available at the forecast origin.
    horizon : int
        Number of future steps to forecast.
    target_col : str
        Target column name.
    window : int
        Number of recent observations to average (default: 3, representing 15 mins).

    Returns
    -------
    pd.Series
        Predicted values for the forecast horizon.
    """
    if train.empty:
        raise ValueError("Training data cannot be empty.")
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    k = min(window, len(train))
    ma_val = float(train[target_col].iloc[-k:].mean())
    preds = np.full(horizon, ma_val, dtype=float)
    return enforce_constraints(preds, target_col)


def mean_reverting_forecast(
    train: pd.DataFrame,
    horizon: int,
    target_col: str,
    phi: float = 0.90,
) -> pd.Series:
    """Forecast using a mean-reversion forecasting rule with a fixed decay factor.

    Formula:
        y_hat[t + h] = mu + (phi ^ h) * (y[t] - mu)

    Explanation:
        - mu is the historical sample mean computed from the training window.
        - y[t] is the last observed target value at the forecast origin.
        - phi is a fixed decay factor (set to 0.90, not fitted via optimization).
        - For the first few steps (small h), phi^h is close to 1, keeping the
          forecast near the latest observed value y[t].
        - Over longer horizons (as h approaches 24 steps = 2 hours), phi^h decreases,
          smoothly pulling the forecast back toward the historical training mean mu.

    Parameters
    ----------
    train : pd.DataFrame
        Training observations available up to the forecast origin.
    horizon : int
        Number of future steps to forecast.
    target_col : str
        Target column name.
    phi : float
        Fixed decay factor in (0, 1) (default: 0.90).

    Returns
    -------
    pd.Series
        Predicted values for the forecast horizon.
    """
    if train.empty:
        raise ValueError("Training data cannot be empty.")
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    history = train[target_col].astype(float)
    mu = float(history.mean())
    last_val = float(history.iloc[-1])

    # Ensure decay rate stays within valid bounds [0, 1]
    decay = float(np.clip(phi, 0.0, 1.0))

    steps = np.arange(1, horizon + 1, dtype=float)
    preds = mu + (decay**steps) * (last_val - mu)

    return enforce_constraints(preds, target_col)


def enforce_constraints(predictions: np.ndarray, target_col: str) -> pd.Series:
    """Enforce physical domain constraints on predicted values.

    - active_vm_count: must be non-negative (>= 0).
    - aggregate_cpu_utilization: must be bounded between 0% and 100%.
    """
    bounded = predictions.copy()
    if target_col == "active_vm_count":
        bounded = np.maximum(0.0, bounded)
    elif target_col == "aggregate_cpu_utilization":
        bounded = np.clip(bounded, 0.0, 100.0)

    return pd.Series(bounded, dtype=float, name=target_col)
