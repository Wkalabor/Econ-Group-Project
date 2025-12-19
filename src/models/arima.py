from __future__ import annotations

import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


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

    def _fit(maxiter: int = 200):
        return ARIMA(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": maxiter})

    converged = True
    fit_warning: str | None = None

    try:
        # Treat convergence warnings as errors so we can retry with a higher maxiter.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = _fit(maxiter=200)
    except ConvergenceWarning as cw:
        converged = False
        fit_warning = str(cw)
        # Retry with a higher iteration cap and silence further convergence warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = _fit(maxiter=500)
    except Exception as exc:
        converged = False
        fit_warning = str(exc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = _fit(maxiter=500)

    def forecast(steps: int = 30) -> pd.DataFrame:
        results = model.get_forecast(steps=steps)
        return results.summary_frame()

    return {"model": model, "forecast": forecast, "converged": converged, "fit_warning": fit_warning}
