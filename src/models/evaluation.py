# ============================================================
# Forecast Evaluation Utilities (FIN41660 Crisis Forecaster)
# ------------------------------------------------------------
# Purpose:
#   Provide reusable helpers for time-series evaluation, including:
#     - Chronological train/test splitting (no shuffling)
#     - Forecast error metrics (RMSE, MAE) with index alignment
#     - Rolling-origin (expanding window) ARIMA evaluation
#
# Output:
#   rolling_origin_arima_evaluation returns:
#     - metrics: RMSE, MAE, and evaluation window dates
#     - predictions: one-step-ahead forecasts aligned to test dates
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

# Import ARIMA selection and estimation utilities from the local module
from .arima import run_arima, select_arima


def time_series_train_test_split(series: pd.Series, test_size: int | float) -> tuple[pd.Series, pd.Series]:
    """
    Chronologically split a series into train and test segments.

    test_size can be an integer count or a float fraction (0,1).
    """
    # Drop missing values so the split is based only on valid observations
    series = series.dropna()

    # Allow test_size to be provided as a fraction of the full sample
    if isinstance(test_size, float):
        # Ensure the fraction is valid (must be strictly between 0 and 1)
        if not 0 < test_size < 1:
            raise ValueError("test_size as float must be between 0 and 1.")
        # Convert fraction to a number of observations, ensuring at least 1 test point
        test_len = max(int(len(series) * test_size), 1)
    else:
        # If an integer is provided, interpret it as the number of test observations
        test_len = int(test_size)

    # Validate the test length
    if test_len <= 0:
        raise ValueError("test_size must be positive.")
    if test_len >= len(series):
        raise ValueError("test_size must be smaller than the length of the series.")

    # Split chronologically: train = early portion, test = last portion
    train = series.iloc[:-test_len]
    test = series.iloc[-test_len:]
    return train, test


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error with alignment on the common index."""
    # Align series on the intersection of dates to avoid mismatched indexing
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")

    # If there is no overlap, metric computation is not meaningful
    if aligned_true.empty:
        raise ValueError("No overlapping index between y_true and y_pred.")

    # RMSE = sqrt(mean((error)^2))
    return float(np.sqrt(np.mean((aligned_true - aligned_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error with alignment on the common index."""
    # Align series on the intersection of dates to avoid mismatched indexing
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")

    # If there is no overlap, metric computation is not meaningful
    if aligned_true.empty:
        raise ValueError("No overlapping index between y_true and y_pred.")

    # MAE = mean(|error|)
    return float(np.mean(np.abs(aligned_true - aligned_pred)))


def _extract_mean_forecast(forecast_df: pd.DataFrame, offset: int = 0) -> float:
    """
    Extract the mean forecast value from a statsmodels forecast summary frame.

    statsmodels commonly uses "mean" or "predicted_mean" for the point forecast,
    but this helper falls back to the first column if needed.
    """
    # Prefer standard column names for the point forecast
    for col in ("mean", "predicted_mean"):
        if col in forecast_df.columns:
            return float(forecast_df[col].iloc[offset])

    # Fallback: use the first column if expected names are not present
    return float(forecast_df.iloc[offset, 0])


def rolling_origin_arima_evaluation(
    series: pd.Series,
    order: tuple[int, int, int] | None = None,
    test_size: int = 30,
    forecast_horizon: int = 1,
) -> dict:
    """
    Perform rolling-origin evaluation with one-step-ahead ARIMA forecasts.

    The ARIMA order can be supplied; otherwise, it is reselected on each iteration.
    """
    # Forecast horizon must be strictly positive
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    # Drop missing values before evaluation
    series = series.dropna()

    # Split into training and test sets (chronological split)
    train, test = time_series_train_test_split(series, test_size=test_size)

    # Store forecasts produced at each step
    preds: list[float] = []

    # Rolling-origin evaluation (expanding window):
    # At each step i, refit ARIMA using all data available up to that point,
    # then forecast ahead by forecast_horizon.
    for i in range(len(test)):
        # History includes full train plus the first i test observations
        history = pd.concat([train, test.iloc[:i]])

        # Use fixed order if provided; otherwise reselect order for each history window
        eval_order = order if order is not None else select_arima(history)

        # Fit the ARIMA model on the current history window
        arima_res = run_arima(history, eval_order)

        # Produce an h-step-ahead forecast (default is one-step-ahead)
        forecast_df = arima_res["forecast"](steps=forecast_horizon)

        # Extract the mean point forecast for the desired horizon step
        preds.append(_extract_mean_forecast(forecast_df, offset=forecast_horizon - 1))

    # Convert forecasts into a Series indexed by the test dates
    predictions = pd.Series(preds, index=test.index, name="Forecast")

    # Compute error metrics and record the evaluation window boundaries
    metrics = {
        "rmse": rmse(test, predictions),
        "mae": mae(test, predictions),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
    }

    # Return both summary metrics and the full prediction series
    return {"metrics": metrics, "predictions": predictions}

# comments by chatgpt edited by group ....
