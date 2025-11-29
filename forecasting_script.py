import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath("."))

from src.models.evaluation import rolling_origin_arima_evaluation
from src.models.arima import run_arima, select_arima
from src.models.garch import run_garch
from src.models.ols import diagnostics_ols, run_ols


REPORT_DIR = Path("report")
DATA_PATH = Path("data") / "fsi.csv"
FORECAST_HORIZON = 30


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/fsi.csv not found. Generate it with load_all_data first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").dropna()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # OLS
    ols_summary_path = REPORT_DIR / "ols_summary.txt"
    try:
        ols_res = run_ols(df)
        diag = diagnostics_ols(ols_res["residuals"])
        with open(ols_summary_path, "w") as fh:
            fh.write(ols_res["model"].summary().as_text())
            fh.write("\n\nDiagnostics:\n")
            for k, v in diag.items():
                fh.write(f"{k}: {v}\n")
        print(f"OLS completed. R2={ols_res['r2']:.3f}. Summary saved to {ols_summary_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"OLS estimation failed: {exc}")

    # ARIMA
    arima_forecast_path = REPORT_DIR / "arima_forecast.csv"
    try:
        order = select_arima(df["FSI"])
        arima_res = run_arima(df["FSI"], order)
        arima_forecast = arima_res["forecast"](FORECAST_HORIZON)
        arima_forecast.to_csv(arima_forecast_path, index_label="Date")
        print(f"ARIMA order {order} forecast saved to {arima_forecast_path}")

        eval_res = rolling_origin_arima_evaluation(
            df["FSI"], order=order, test_size=FORECAST_HORIZON, forecast_horizon=1
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

    # GARCH
    garch_forecast_path = REPORT_DIR / "garch_vol_forecast.csv"
    try:
        garch_res = run_garch(df["FSI"])
        garch_forecast = garch_res["forecast"](FORECAST_HORIZON)
        garch_forecast.to_csv(garch_forecast_path)
        print(f"GARCH volatility forecast saved to {garch_forecast_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"GARCH estimation failed: {exc}")


if __name__ == "__main__":
    main()
