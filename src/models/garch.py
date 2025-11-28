from __future__ import annotations

import pandas as pd
from arch import arch_model


def run_garch(series: pd.Series, p: int = 1, q: int = 1) -> dict:
    """Fit a basic GARCH(p, q) model on the provided series (mean zero)."""
    series = series.dropna()
    demeaned = series - series.mean()
    model = arch_model(demeaned, mean="Zero", vol="GARCH", p=p, q=q, rescale=True)
    result = model.fit(disp="off")

    def forecast(horizon: int = 30) -> pd.DataFrame:
        forecast_res = result.forecast(horizon=horizon)
        variance = forecast_res.variance.iloc[-1]
        volatility = variance.pow(0.5)
        volatility.name = "volatility"
        volatility.index.name = "step"
        return volatility.to_frame()

    return {"model": result, "forecast": forecast}
