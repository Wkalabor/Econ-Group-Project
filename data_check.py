# ============================================================
# Financial Stress Index (FSI) sanity check and visualisation
# ============================================================
# This script loads the constructed Financial Stress Index (FSI),
# prints basic diagnostics to the console, and saves a simple
# time-series plot for verification and reporting purposes.
# ============================================================

from pathlib import Path  # Used for robust, OS-independent file path handling

import matplotlib.pyplot as plt  # Plotting library for visual inspection of the FSI
import pandas as pd  # Data handling and analysis library


def main() -> None:
    # Resolve the root directory of the repository based on this file's location
    repo_root = Path(__file__).resolve().parent

    # Define the expected path to the cleaned FSI CSV file
    data_path = repo_root / "data" / "fsi.csv"

    # Check that the FSI data exists; if not, instruct the user how to generate it
    if not data_path.exists():
        raise FileNotFoundError("data/fsi.csv not found. Run src.utils.load_data.load_all_data first.")

    # Load the FSI data, parsing the Date column and setting it as the index
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    # Print the first few rows to verify structure and recent data
    print("FSI head:")
    print(df.head())

    # Print the last few rows to verify the most recent observations
    print("\nFSI tail:")
    print(df.tail())

    # Print summary statistics to check scale, dispersion, and basic moments
    print("\nFSI describe():")
    print(df.describe())

    # Create a time-series plot of the Financial Stress Index
    plt.figure(figsize=(10, 5))
    df["FSI"].plot()

    # Add a descriptive title to the plot
    plt.title("Financial Stress Index")

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Define output path for the saved figure
    output_path = repo_root / "report" / "fsi_check.png"

    # Ensure the report directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure to disk for inclusion in reports or debugging
    plt.savefig(output_path)

    # Confirm to the user where the file has been saved
    print(f"\nSaved plot to {output_path}")


# Standard Python entry point guard
if __name__ == "__main__":
    main()

# Comments by ChatGPT, edited by group
