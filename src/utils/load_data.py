import pandas as pd
import yfinance as yf


def _extract_price(data: pd.DataFrame, field: str) -> pd.Series:
    """Handle both flat and MultiIndex columns from yfinance downloads."""
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if field in level0:
            return data.xs(field, level=0, axis=1).squeeze()
    elif field in data.columns:
        return data[field]
    raise KeyError(f"{field} not found in downloaded VIX data")


def load_vix(start: str = "2005-01-01", end: str | None = None) -> pd.Series:
    """Download daily VIX data and return a Series named 'VIX' indexed by date."""
    data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    series = None
    for field in ("Adj Close", "Close"):
        try:
            series = _extract_price(data, field)
            break
        except KeyError:
            continue
    if series is None:
        raise KeyError("No Adj Close or Close column found for VIX")
    series = series.rename("VIX")
    series.index.name = "Date"
    return series


def load_all_data(start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Placeholder to aggregate multiple stress indicators.

    Currently returns a single-column DataFrame with VIX data.
    """
    vix = load_vix(start=start, end=end)
    return pd.DataFrame({"VIX": vix})
