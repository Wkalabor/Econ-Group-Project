# ============================================================
# VIX Data Sanity Check Script
# ------------------------------------------------------------
# Purpose:
#   Load the VIX time series using the project data loader and
#   print basic summaries (head, tail, descriptive statistics)
#   to verify data integrity and structure.
# ============================================================

import os
import sys

# Ensure the project root is on the Python path so that
# `src.utils.load_data` can be imported when running this file directly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the VIX data loading function from project utilities
from src.utils.load_data import load_vix


def main() -> None:
    """
    Load the VIX dataset and print basic diagnostic outputs
    for quick inspection.
    """
    # Load VIX data (expected to be a pandas Series or DataFrame)
    vix = load_vix()

    # Display the first few observations
    print("VIX head:")
    print(vix.head())

    # Display the last few observations
    print("\nVIX tail:")
    print(vix.tail())

    # Display descriptive statistics (count, mean, std, etc.)
    print("\nVIX describe():")
    print(vix.describe())


# Standard Python entry point guard
# Ensures main() only runs when this file is executed directly
if __name__ == "__main__":
    main()

# comments by chatgpt edited by group ....
