# ============================================================
# Unit Tests: Evaluation Utilities (Train/Test Split, RMSE, MAE)
# ------------------------------------------------------------
# Purpose:
#   Validate the correctness of helper functions used in
#   forecast evaluation, including:
#     - Time-series train/test splitting
#     - Root Mean Squared Error (RMSE)
#     - Mean Absolute Error (MAE)
#
# These tests ensure that evaluation metrics and data splitting
# behave as expected before being used in the Crisis Forecaster
# pipeline.
# ============================================================

from __future__ import annotations

import math  # Used for square root and numerical comparisons

import pandas as pd
import pytest

# Skip these tests entirely if pmdarima is not installed
# (keeps the test suite robust across environments)
pytest.importorskip("pmdarima")

# Import the evaluation utilities under test
from src.models.evaluation import mae, rmse, time_series_train_test_split


def test_time_series_train_test_split_preserves_order_and_sizes() -> None:
    """
    Test that the time-series train/test split:
      - Preserves chronological order
      - Produces correct train and test sizes
      - Does not shuffle observations
    """
    # Create a simple deterministic time series
    n = 50
    series = pd.Series(range(n), index=pd.date_range("2020-01-01", periods=n, freq="D"))

    # Define test size as a fraction of the full sample
    train_ratio = 0.8
    test_size = 1 - train_ratio

    # Perform the split
    train, test = time_series_train_test_split(series, test_size=test_size)

    # Expected sizes based on the implementation logic
    expected_test_len = max(int(n * test_size), 1)
    expected_train_len = n - expected_test_len

    # Check that lengths match expectations
    assert len(train) == expected_train_len
    assert len(test) == expected_test_len

    # Ensure both train and test indices are ordered in time
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing

    # Ensure train data strictly precedes test data
    assert train.index.max() < test.index.min()

    # Ensure no observations are lost or reordered
    assert list(train.index) + list(test.index) == list(series.index)


def test_rmse_and_mae_known_values() -> None:
    """
    Test RMSE and MAE against hand-calculated values
    using a small example with known errors.
    """
    # Define simple true and predicted series
    y_true = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 1, 4])

    # Manually compute expected metric values
    expected_rmse = math.sqrt((0**2 + 1**2 + 1**2) / 3)
    expected_mae = (0 + 1 + 1) / 3

    # Verify RMSE calculation
    assert math.isclose(rmse(y_true, y_pred), expected_rmse, rel_tol=1e-9)

    # Verify MAE calculation
    assert math.isclose(mae(y_true, y_pred), expected_mae, rel_tol=1e-9)

# comments by chatgpt edited by group ....
