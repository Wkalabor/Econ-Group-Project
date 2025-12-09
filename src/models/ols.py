from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt


def run_ols(df: pd.DataFrame) -> dict:
    """
    Estimate a linear model: FSI ~ VIX + Spread + CDS.
    Returns a dict containing the fitted model, residuals, R², and parameters.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    y = df["FSI"].dropna()
    X = df.loc[y.index, ["VIX", "Spread", "CDS"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # ✅ Save summary to file for reproducibility
    with open("report/ols_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    return {
        "model": model,
        "residuals": model.resid,
        "r2": model.rsquared,
        "params": model.params,
        "tvalues": model.tvalues,
        "pvalues": model.pvalues,
    }


def diagnostics_ols(residuals: pd.Series, lags: int = 12) -> dict:
    """
    Run Ljung–Box, ARCH LM, and Jarque–Bera diagnostics on residuals.
    Returns key statistics and p-values.
    """
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    arch_stat, arch_pvalue, _, _ = het_arch(residuals, nlags=lags)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

    return {
        "ljung_box_stat": lb["lb_stat"].iloc[0],
        "ljung_box_pvalue": lb["lb_pvalue"].iloc[0],
        "arch_lm_stat": arch_stat,
        "arch_lm_pvalue": arch_pvalue,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "skew": skew,
        "kurtosis": kurtosis,
    }

def plot_residuals(residuals: pd.Series, output_path: str = "report/fsi_check.png"):
    """Plot OLS residuals and save to report folder."""
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, color="steelblue")
    plt.title("OLS Residuals – FSI ~ VIX + Spread + CDS")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
