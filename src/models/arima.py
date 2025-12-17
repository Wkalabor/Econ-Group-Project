from __future__ import annotations

import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA


def select_arima(series: pd.Series, order_max: tuple[int, int, int] = (3, 2, 3)) -> tuple[int, int, int]:
    series = series.dropna()
    model = pm.auto_arima(
        series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=order_max[0],
        max_d=order_max[1],
        max_q=order_max[2],
    )
    return model.order


def run_arima(series: pd.Series, order: tuple[int, int, int]) -> dict:
    series = series.dropna()

    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method_kwargs={"maxiter": 200})

    def forecast(steps: int = 30) -> pd.DataFrame:
        results = model.get_forecast(steps=steps)
        return results.summary_frame()

    return {"model": model, "forecast": forecast}

