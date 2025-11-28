from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def run_ols(df: pd.DataFrame) -> dict:
    """
    Estimate a linear model: FSI ~ VIX + Spread + CDS.

    Returns a dict containing the fitted model, residuals, r-squared, and parameters.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    y = df["FSI"].dropna()
    X = df.loc[y.index, ["VIX", "Spread", "CDS"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return {
        "model": model,
        "residuals": model.resid,
        "r2": model.rsquared,
        "params": model.params,
    }


def diagnostics_ols(residuals: pd.Series, lags: int = 12) -> dict:
    """Run simple residual diagnostics and return stats/p-values."""
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_stat = lb["lb_stat"].iloc[0]
    lb_pvalue = lb["lb_pvalue"].iloc[0]

    arch_stat, arch_pvalue, _, _ = het_arch(residuals, nlags=lags)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

    return {
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "arch_lm_stat": arch_stat,
        "arch_lm_pvalue": arch_pvalue,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "skew": skew,
        "kurtosis": kurtosis,
    }
