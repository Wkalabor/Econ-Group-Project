# ============================================================
# Unit Test: load_all_data() basic integrity and structure
# ------------------------------------------------------------
# Purpose:
#   Verify that the load_all_data utility returns a DataFrame
#   with the expected columns, index type, and basic data
#   cleanliness properties required for the Crisis Forecaster
#   project.
# ============================================================

from __future__ import annotations

import pandas as pd
import pytest

# Import the data loading utility under test
from src.utils.load_data import load_all_data


def test_load_all_data_basic_shape() -> None:
    """
    Test that load_all_data returns a well-formed DataFrame with
    required columns, a DatetimeIndex, and no missing values in
    the FSI series after initial observations.
    """
    # Attempt to load the full dataset; skip the test if external
    # data access is unavailable (e.g. network issues)
    try:
        df = load_all_data(save_csv=False)
    except Exception as exc:  # pragma: no cover - network guard
        pytest.skip(f"Data download unavailable: {exc}")

    # Collect column names for flexible membership checks
    cols = set(df.columns)

    # Core variables must be present
    assert "VIX" in cols
    assert "FSI" in cols

    # Allow for alternative naming conventions across datasets
    assert any(col in cols for col in ["Spread", "CreditSpread"])
    assert any(col in cols for col in ["CDS", "BankRisk"])

    # Index should be time-based for time-series analysis
    assert isinstance(df.index, pd.DatetimeIndex)

    # Dataset should have a minimum reasonable length
    assert len(df) > 10

    # Drop initial observations and confirm FSI contains no NaNs
    fsi_clean = df["FSI"].iloc[5:]
    assert not fsi_clean.isna().any()

# comments by chatgpt edited by group ....
