# ============================================================
# Standalone Forecasting and Evaluation Pipeline
# FIN41660 Crisis Forecaster Project
#
# This script runs the full empirical pipeline for the project:
#  - Loads the Financial Stress Index (FSI)
#  - Estimates OLS, ARIMA, and GARCH models
#  - Produces forecasts and rolling-origin evaluation metrics
#  - Saves figures and tables for inclusion in the written report
# ============================================================

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Ensure the project root is on the Python path so src/ imports work
sys.path.append(os.path.abspath("."))

# Import model and evaluation utilities from the project
from src.models.evaluation import rolling_origin_arima_evaluation
from src.models.arima import run_arima, select_arima
from src.models.garch import run_garch
from src.models.ols import diagnostics_ols, plot_residuals, run_ols

# Use a non-interactive backend so figures can be saved without a display
matplotlib.use("Agg")


"""
Standalone pipeline for the FIN41660 Crisis Forecaster project.

Runs OLS, ARIMA, and GARCH on the FSI, writes forecasts/metrics, and saves
figures for the written report.
"""

# Directory where figures and tables for the report will be saved
REPORT_DIR = Path("report")

# Path to the cleaned FSI dataset
DATA_PATH = Path("data") / "fsi.csv"

# Forecast horizon (number of days ahead)
FORECAST_HORIZON = 30


def _ensure_datetime_index(series: pd.Series, start: pd.Timestamp, periods: int) -> pd.Series:
    """
    Guarantee a datetime index for forecast series when statsmodels
    returns an integer-based index.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    future_index = pd.date_range(start + pd.Timedelta(days=1), periods=periods, freq="D")
    series.index = future_index
    return series


def save_figures(
    df: pd.DataFrame,
    arima_forecast: pd.DataFrame | None,
    garch_forecast: pd.DataFrame | None,
    garch_model,
) -> None:
    """
    Save key figures used in the written report, including:
    - Full FSI history
    - ARIMA forecast with confidence intervals
    - GARCH conditional volatility and forecast
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- FSI time-series plot ---
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["FSI"].plot(ax=ax, label="FSI")
        ax.set_title("Financial Stress Index (full history)")
        ax.set_ylabel("Index level")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "fsi_series.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"FSI figure failed: {exc}")

    # --- ARIMA forecast plot ---
    try:
        if arima_forecast is not None:
            history = df["FSI"].iloc[-200:]
            forecast_mean = arima_forecast["mean"].copy()
            forecast_mean = _ensure_datetime_index(
                forecast_mean, df.index.max(), len(forecast_mean)
            )
            arima_forecast = arima_forecast.copy()
            arima_forecast.index = forecast_mean.index

            fig, ax = plt.subplots(figsize=(8, 4))
            history.plot(ax=ax, label="Historical FSI")
            forecast_mean.plot(ax=ax, label="ARIMA forecast")
            ax.fill_between(
                arima_forecast.index,
                arima_forecast["mean_ci_lower"],
                arima_forecast["mean_ci_upper"],
                alpha=0.2,
                label="95% CI",
            )
            ax.set_title("ARIMA forecast with 95% confidence interval")
            ax.legend()
            fig.tight_layout()
            fig.savefig(REPORT_DIR / "arima_forecast.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ARIMA figure failed: {exc}")

    # --- GARCH volatility plot ---
    try:
        if garch_forecast is not None and garch_model is not None:
            cond_vol = pd.Series(
                garch_model.conditional_volatility, index=df.index, name="Cond vol"
            )
            forecast_vol = garch_forecast["volatility"].copy()
            if not isinstance(forecast_vol.index, pd.DatetimeIndex):
                future_index = pd.date_range(
                    df.index.max() + pd.Timedelta(days=1),
                    periods=len(forecast_vol),
                    freq="D",
                )
                forecast_vol.index = future_index

            fig, ax = plt.subplots(figsize=(8, 4))
            cond_vol.iloc[-250:].plot(ax=ax, label="In-sample volatility")
            forecast_vol.plot(ax=ax, label="Forecast volatility")
            ax.set_title("GARCH conditional volatility and forecast")
            ax.set_ylabel("Volatility")
            ax.legend()
            fig.tight_layout()
            fig.savefig(REPORT_DIR / "garch_vol_forecast.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"GARCH figure failed: {exc}")


def main() -> None:
    # Ensure the FSI dataset exists before running the pipeline
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/fsi.csv not found. Generate it with load_all_data first.")

    # Load the FSI dataset and drop missing values
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").dropna()

    # Ensure the report directory exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise placeholders for forecasts and models
    arima_forecast: pd.DataFrame | None = None
    garch_forecast: pd.DataFrame | None = None
    garch_model = None

    # -----------------------------
    # OLS estimation and diagnostics
    # -----------------------------
    ols_summary_path = REPORT_DIR / "ols_summary.txt"
    try:
        ols_res = run_ols(df)
        diag = diagnostics_ols(ols_res["residuals"])

        # Generate and save residual diagnostics plot
        plot_residuals(ols_res["residuals"])

        # Write regression output and diagnostics to file
        with open(ols_summary_path, "w") as fh:
            fh.write(ols_res["model"].summary().as_text())
            fh.write("\n\nDiagnostics:\n")
            for k, v in diag.items():
                fh.write(f"{k}: {v}\n")

        print(f"OLS completed. R2={ols_res['r2']:.3f}. Summary saved to {ols_summary_path}")
    except Exception as exc:
        print(f"OLS estimation failed: {exc}")

    # -----------------------------
    # ARIMA estimation and evaluation
    # -----------------------------
    arima_forecast_path = REPORT_DIR / "arima_forecast.csv"
    try:
        order = select_arima(df["FSI"])
        arima_res = run_arima(df["FSI"], order)
        arima_forecast = arima_res["forecast"](FORECAST_HORIZON)
        arima_forecast.to_csv(arima_forecast_path, index_label="Date")
        print(f"ARIMA order {order} forecast saved to {arima_forecast_path}")

        # Rolling-origin out-of-sample evaluation
        eval_res = rolling_origin_arima_evaluation(
            df["FSI"],
            order=order,
            test_size=FORECAST_HORIZON,
            forecast_horizon=1,
        )
        metrics = eval_res["metrics"]

        metrics_df = pd.DataFrame(
            [
                {
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "train_end_date": metrics["train_end"].date(),
                    "test_start_date": metrics["test_start"].date(),
                    "test_end_date": metrics["test_end"].date(),
                }
            ]
        )
        metrics_path = REPORT_DIR / "forecast_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"ARIMA evaluation metrics saved to {metrics_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ARIMA estimation failed: {exc}")

    # -----------------------------
    # GARCH estimation and forecast
    # -----------------------------
    garch_forecast_path = REPORT_DIR / "garch_vol_forecast.csv"
    try:
        garch_res = run_garch(df["FSI"])
        garch_forecast = garch_res["forecast"](FORECAST_HORIZON)
        garch_model = garch_res["model"]
        garch_forecast.to_csv(garch_forecast_path)
        print(f"GARCH volatility forecast saved to {garch_forecast_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"GARCH estimation failed: {exc}")

    # Save all figures required for the report
    save_figures(df, arima_forecast, garch_forecast, garch_model)


# Execute the full pipeline when run as a script
if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# Comments by ChatGPT, edited by group
# ------------------------------------------------------------
