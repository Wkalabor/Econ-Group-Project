# ============================================================
# FIN41660 Crisis Forecaster — Streamlit Dashboard (app/app.py)
# ------------------------------------------------------------
# Purpose:
#   Interactive dashboard for exploring the Financial Stress Index (FSI)
#   and running a modelling suite (OLS, ARIMA, GARCH) on a selected
#   training window.
#
# Features (tabs):
#   - Home: project overview
#   - Data explorer: view FSI + components and plot time series
#   - Model estimation: run OLS/ARIMA/GARCH and show key outputs
#   - Forecasts and diagnostics: plot ARIMA forecast + confidence bands,
#     plus GARCH volatility forecast
#   - Forecast evaluation: rolling-origin ARIMA evaluation (RMSE/MAE)
#   - Crisis simulation: apply stylised shocks to VIX or Spread and show
#     baseline vs shock forecast paths
#
# Notes:
#   - Uses Streamlit caching for data loading to keep the UI responsive.
#   - Adds repo root to sys.path so `src.*` imports work when run as:
#       streamlit run app/app.py
# ============================================================

from __future__ import annotations
import os
import sys

# Compute the repository root directory (one level above /app)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the repository root is available on sys.path so imports from src/ work
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

# Project model utilities
from src.models.evaluation import rolling_origin_arima_evaluation
from src.models.arima import run_arima, select_arima
from src.models.garch import run_garch
from src.models.ols import diagnostics_ols, run_ols

# Data pipeline utility
from src.utils.load_data import load_all_data




@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
            """Load and cache the FSI dataset."""
            # load_all_data downloads/builds the FSI dataset and returns a DataFrame
            # Streamlit caches this so repeated tab switches do not re-download data
            return load_all_data()


def run_model_suite(df: pd.DataFrame, horizon: int) -> Dict[str, object]:
            """Run OLS, ARIMA, and GARCH models on the provided training data."""
            # Container for model outputs
            results: Dict[str, object] = {}

            # -----------------------------
            # 1) OLS estimation + diagnostics
            # -----------------------------
            ols_res = run_ols(df)
            results["ols"] = {**ols_res, "diagnostics": diagnostics_ols(ols_res["residuals"])}

            # -----------------------------
            # 2) ARIMA: select order + forecast
            # -----------------------------
            order = select_arima(df["FSI"])
            arima_res = run_arima(df["FSI"], order)
            results["arima"] = {"order": order, "forecast": arima_res["forecast"](horizon)}

            # -----------------------------
            # 3) GARCH: fit volatility model + forecast
            # -----------------------------
            garch_res = run_garch(df["FSI"])
            results["garch"] = {"forecast": garch_res["forecast"](horizon)}

            # Return all model results to the Streamlit UI
            return results


def main() -> None:
            # Streamlit page configuration (title + wide layout for plots/tables)
            st.set_page_config(page_title="Crisis Forecaster", layout="wide")
            st.title("Crisis Forecaster prototype")

            # Load data (cached)
            data = get_data()
            min_date = data.index.min().date()
            max_date = data.index.max().date()

            # -----------------------------
            # Sidebar controls
            # -----------------------------
            st.sidebar.header("Controls")

            # Choose whether to show only the FSI or include component series
            view_choice = st.sidebar.radio(
                "Data view", options=["FSI + components", "FSI only"], index=0, horizontal=False
            )

            # Choose the training sample end date (training = start -> train_end)
            train_end: date = st.sidebar.slider(
                "Training sample end date",
                min_value=min_date,
                max_value=max_date,
                value=max_date,
                format="YYYY-MM-DD",
            )

            # Forecast horizon in days (used by ARIMA + GARCH)
            horizon = st.sidebar.slider("Forecast horizon (days)", min_value=5, max_value=60, value=30, step=5)

            # Button to run/refresh models
            run_models = st.sidebar.button("Run models")

            # Slice training data up to the selected end date
            train_df = data.loc[:pd.to_datetime(train_end)]

            # Guard against running models on extremely short samples
            insufficient_data = len(train_df) < 100

            # Initialise session state storage for model outputs
            if "model_results" not in st.session_state:
                st.session_state["model_results"] = None

            results = None
            models_ready = False
            model_error: Exception | None = None

            # -----------------------------
            # Run models when user clicks button
            # -----------------------------
            if run_models and not insufficient_data:
                try:
                    results = run_model_suite(train_df, horizon)

                    # Cache model results in session state so tabs can access them
                    st.session_state["model_results"] = {
                        "results": results,
                        "train_end": train_end,
                        "horizon": horizon,
                    }
                    models_ready = True
                except Exception as exc:  # pragma: no cover - defensive
                    # Store the error so it can be displayed in the UI
                    model_error = exc
                    st.session_state["model_results"] = None

            # -----------------------------
            # If models were run previously, reuse cached results when settings match
            # -----------------------------
            cached_results = st.session_state.get("model_results")
            if not models_ready and cached_results:
                if (
                    cached_results.get("train_end") == train_end
                    and cached_results.get("horizon") == horizon
                    and not insufficient_data
                ):
                    results = cached_results["results"]
                    models_ready = True

            # Create tab layout for navigation
            home_tab, data_tab, model_tab, forecast_tab, eval_tab, sim_tab = st.tabs(
                [
                    "Home",
                    "Data explorer",
                    "Model estimation",
                    "Forecasts and diagnostics",
                    "Forecast evaluation",
                    "Crisis simulation",
                ]
            )

            # ============================================================
            # TAB 1: Home
            # ============================================================
            with home_tab:
                st.subheader("Overview")
                st.write(
                    "Explore the Financial Stress Index (FSI), fit OLS/ARIMA/GARCH models, "
                    "and visualize forecasts. Use the sidebar to choose the training window and "
                    "forecast horizon, then navigate the tabs for data, estimation outputs, diagnostics, "
                    "and evaluation."
                )
                st.write(
                    "The app keeps your selections in the sidebar consistent across tabs so you can move "
                    "from data inspection to model estimation and forecast review without losing context."
                )

            # ============================================================
            # TAB 2: Data explorer
            # ============================================================
            with data_tab:
                st.subheader("Data explorer")

                # Decide which columns to show
                cols_to_show = list(data.columns) if view_choice == "FSI + components" else ["FSI"]

                # Display most recent observations
                st.write("Recent observations")
                st.dataframe(data[cols_to_show].tail(10))

                # Quick line chart of selected series
                st.line_chart(data[cols_to_show])

            # ============================================================
            # TAB 3: Model estimation
            # ============================================================
            with model_tab:
                st.subheader("Model estimation")

                # Sample-size guard message
                if insufficient_data:
                    st.info("Select a longer sample to run models (need at least 100 daily observations).")

                # Display model estimation error (if any)
                elif model_error:
                    st.error(f"Model estimation failed: {model_error}")

                # Display results if models ran successfully
                elif models_ready and results is not None:
                    st.write(f"Training sample ends on {train_end} | Forecast horizon: {horizon} days")

                    # -------- OLS outputs --------
                    st.markdown("**OLS results**")
                    ols = results["ols"]
                    st.write(f"R-squared: {ols['r2']:.3f}")
                    st.dataframe(ols["params"].to_frame("coef"))

                    # Residual diagnostic outputs formatted for readability
                    st.write("Diagnostics")
                    diag_df = pd.Series(ols["diagnostics"]).apply(lambda x: f"{x:.4g}").to_frame("value")
                    st.dataframe(diag_df)

                    # -------- ARIMA outputs --------
                    st.markdown("**ARIMA selection**")
                    st.write(f"Selected order: {results['arima']['order']}")

                    # Optional warning if ARIMA fit was flagged as not converged
                    if not results["arima"].get("converged", True):
                        st.warning(
                            "ARIMA fit did not fully converge; forecasts may be less reliable. "
                            "Order selection succeeded but optimization hit its iteration limit."
                        )

                    # -------- GARCH outputs --------
                    st.markdown("**GARCH**")
                    st.write("Volatility forecast prepared.")

                else:
                    st.info("Use the sidebar to run models and view estimation outputs.")

            # ============================================================
            # TAB 4: Forecasts and diagnostics
            # ============================================================
            with forecast_tab:
                st.subheader("Forecasts and diagnostics")

                # Require that models have been run
                if not models_ready or results is None:
                    st.info("Run models from the sidebar to view forecasts and diagnostics.")
                else:
                    # Pull recent history window for plotting
                    history = train_df["FSI"].iloc[-200:]

                    # ARIMA forecast summary frame
                    arima_forecast = results["arima"]["forecast"]
                    forecast_mean = arima_forecast["mean"]

                    # Keep this combined variable (used for quick checks / optional plots)
                    combined = pd.concat([history, forecast_mean.rename("forecast")])

                    # --- Historical FSI (same style as forecast plot) ---
                    fig0, ax0 = plt.subplots(figsize=(7, 3.5))
                    ax0.plot(history.index, history.values, label="Historical FSI")
                    ax0.set_title("FSI: historical series (training sample)")
                    ax0.set_xlabel("Date")
                    ax0.set_ylabel("FSI")
                    ax0.legend()
                    ax0.grid(True, alpha=0.2)
                    st.pyplot(fig0, use_container_width=False)

                    # --- ARIMA forecast with confidence interval ---
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.plot(history.index, history.values, label="Historical FSI")
                    ax.plot(forecast_mean.index, forecast_mean.values, label="ARIMA forecast")
                    ax.fill_between(
                        forecast_mean.index,
                        arima_forecast["mean_ci_lower"],
                        arima_forecast["mean_ci_upper"],
                        alpha=0.25,
                        label="95% CI",
                    )
                    ax.set_title("FSI: ARIMA forecast with 95% confidence interval")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("FSI")
                    ax.legend()
                    st.pyplot(fig, use_container_width=False)

                    # --- GARCH volatility forecast ---
                    garch_forecast = results["garch"]["forecast"]

                    # Ensure forecast is a DataFrame with a "volatility" column for plotting
                    if isinstance(garch_forecast, pd.Series):
                        garch_forecast = garch_forecast.to_frame("volatility")
                    else:
                        if garch_forecast.shape[1] == 1:
                            garch_forecast.columns = ["volatility"]

                    # Replace the index with 1..H so it reads as a forecast step chart
                    garch_plot = garch_forecast.copy()
                    garch_plot.index = range(1, len(garch_plot) + 1)

                    fig2, ax2 = plt.subplots(figsize=(7, 2.8))
                    ax2.plot(garch_plot.index, garch_plot.iloc[:, 0].values)
                    ax2.set_title("GARCH: forecasted conditional volatility (next H steps)")
                    ax2.set_xlabel("Forecast step (days ahead)")
                    ax2.set_ylabel("Volatility")
                    ax2.grid(True, alpha=0.2)
                    st.pyplot(fig2, use_container_width=False)

            # ============================================================
            # TAB 5: Forecast evaluation
            # ============================================================
            with eval_tab:
                st.subheader("Forecast evaluation")
                st.write(
                    "Out-of-sample performance for the ARIMA model on the Financial Stress Index. "
                    "Choose a training ratio and run the evaluation to see forecast accuracy on the held-out window."
                )

                # Cap test window to keep Streamlit responsive (rolling refits can be slow)
                MAX_EVAL_POINTS = 90  # cap test window to keep runtime reasonable

                # ---- user controls ----
                train_ratio = st.slider(
                    "Training ratio",
                    min_value=0.6,
                    max_value=0.9,
                    value=0.8,
                    step=0.05,
                    key="eval_train_ratio",
                )

                # Choose a fixed ARIMA order for evaluation (avoids repeated auto-selection)
                order_choice = st.selectbox(
                    "ARIMA order (for evaluation)",
                    options=[(1, 0, 1), (1, 1, 1), (2, 0, 1), (2, 1, 2)],
                    index=0,
                    key="eval_order_choice",
                )

                # Use full dataset for evaluation (not limited to training slice)
                eval_df = get_data()
                fsi_series = eval_df["FSI"]

                # Compute test size from the chosen train_ratio
                test_fraction = 1 - train_ratio
                requested_test_size = int(len(fsi_series) * test_fraction)

                # Apply cap so the evaluation doesn't take too long
                test_size = max(1, min(requested_test_size, MAX_EVAL_POINTS))

                # Inform user when the test window is capped
                if requested_test_size > MAX_EVAL_POINTS:
                    st.caption(
                        f"Test window capped at {MAX_EVAL_POINTS} observations (requested {requested_test_size}) "
                        "to keep evaluation responsive."
                    )

                # Run evaluation button
                run_eval = st.button("Run evaluation", key="eval_run_btn")

                # ---- session cache so results persist without rerunning ----
                if "eval_results" not in st.session_state:
                    st.session_state["eval_results"] = None

                # Run evaluation when clicked
                if run_eval:
                    try:
                        with st.spinner("Running rolling-origin ARIMA evaluation... this may take a moment"):
                            eval_res = rolling_origin_arima_evaluation(
                                fsi_series,
                                order=order_choice,
                                test_size=test_size,
                                forecast_horizon=1,
                            )

                            # Save results so tab switches don't rerun expensive computation
                            st.session_state["eval_results"] = {
                                "train_ratio": train_ratio,
                                "order": order_choice,
                                "test_size": test_size,
                                "eval_res": eval_res,
                            }

                    except Exception as exc:
                        st.session_state["eval_results"] = None
                        st.error(f"Evaluation failed: {exc}")

                # ---- display cached results if available ----
                cached = st.session_state.get("eval_results")

                if cached is None:
                    st.info("Click **Run evaluation** to compute out-of-sample performance.")
                else:
                    # If settings changed, warn user that cached results are stale
                    if (
                        cached["train_ratio"] != train_ratio
                        or cached["order"] != order_choice
                        or cached.get("test_size") != test_size
                    ):
                        st.warning("Settings changed. Click **Run evaluation** again to update results.")
                    else:
                        eval_res = cached["eval_res"]
                        metrics = eval_res["metrics"]
                        preds = eval_res["predictions"]

                        # Align actual vs predicted for plotting and error computation
                        actual = fsi_series.loc[preds.index]
                        actual_aligned, preds_aligned = actual.align(preds, join="inner")

                        if actual_aligned.empty:
                            st.warning("No overlapping dates between actual and predicted series.")
                        else:
                            # Summary metrics
                            col1, col2 = st.columns(2)
                            col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                            col2.metric("MAE", f"{metrics['mae']:.4f}")

                            # Evaluation window dates
                            train_end_dt = pd.to_datetime(metrics["train_end"]).date()
                            test_start_dt = pd.to_datetime(metrics["test_start"]).date()
                            test_end_dt = pd.to_datetime(metrics["test_end"]).date()
                            st.write(
                                f"Train end: {train_end_dt} | Test window: {test_start_dt} to {test_end_dt}"
                            )

                            # Additional note about the capped evaluation window
                            st.caption(
                                f"Using {len(fsi_series) - cached['test_size']} training observations "
                                f"and {cached['test_size']} test observations (capped at {MAX_EVAL_POINTS})."
                            )

                            # Plot actual vs predicted
                            overlay_df = pd.DataFrame(
                                {"Actual FSI": actual_aligned, "Predicted FSI": preds_aligned}
                            )
                            st.line_chart(overlay_df)

                            # Plot forecast errors
                            errors = (actual_aligned - preds_aligned).rename("Forecast error")
                            st.line_chart(errors.to_frame())

                            # Show last few rows in a table
                            st.dataframe(
                                pd.DataFrame({"Actual": actual_aligned, "Predicted": preds_aligned}).tail(10)
                            )



            # ============================================================
            # TAB 6: Crisis simulation
            # ============================================================
            with sim_tab:
                st.subheader("Crisis simulation")
                st.write(
                    "Simulate simple shock scenarios to VIX or credit spreads and view their impact on the "
                    "FSI forecast path. Shocks are applied as constant shifts using OLS coefficients."
                )

                # Define available shock scenarios
                scenario_options = [
                    "None",
                    "VIX shock plus 25 percent",
                    "VIX shock plus 50 percent",
                    "Credit spread shock plus 50bp",
                    "Credit spread shock plus 100bp",
                ]

                # Choose which scenario to apply
                scenario_choice = st.selectbox("Shock scenario", options=scenario_options, index=0)

                # Choose simulation horizon (days)
                sim_horizon = st.slider(
                    "Scenario horizon (days)", min_value=5, max_value=60, value=horizon, step=5
                )

                # Run simulation button
                if st.button("Run simulation"):
                    if insufficient_data:
                        st.info("Select a longer sample to run simulations (need at least 100 daily observations).")
                    else:
                        try:
                            sim_train = train_df

                            # If OLS already ran, reuse parameters; otherwise refit OLS quickly
                            ols_params = None
                            if models_ready and results is not None:
                                ols_params = results["ols"]["params"]
                            if ols_params is None:
                                ols_params = run_ols(sim_train)["params"]

                            # Extract coefficients for shock transmission
                            vix_beta = float(ols_params.get("VIX", 0.0))
                            spread_beta = float(ols_params.get("Spread", 0.0))

                            # Use the most recent observed values as the "current" baseline
                            latest_obs = sim_train.iloc[-1]
                            current_vix = float(latest_obs["VIX"])
                            current_spread = float(latest_obs["Spread"])

                            # Initialise shift and descriptive text
                            shock_shift = 0.0
                            shock_desc = "No shock applied."

                            # Apply the selected scenario as a constant shift in FSI using OLS betas
                            if scenario_choice == "VIX shock plus 25 percent":
                                shock_size = 0.25 * current_vix
                                shock_shift = vix_beta * shock_size
                                shock_desc = (
                                    f"VIX increased by 25% of current level ({shock_size:.2f}); "
                                    f"FSI shift = beta_VIX ({vix_beta:.4f}) * shock."
                                )
                            elif scenario_choice == "VIX shock plus 50 percent":
                                shock_size = 0.50 * current_vix
                                shock_shift = vix_beta * shock_size
                                shock_desc = (
                                    f"VIX increased by 50% of current level ({shock_size:.2f}); "
                                    f"FSI shift = beta_VIX ({vix_beta:.4f}) * shock."
                                )
                            elif scenario_choice == "Credit spread shock plus 50bp":
                                shock_size = 0.50  # 50 basis points expressed in spread units
                                shock_shift = spread_beta * shock_size
                                shock_desc = (
                                    f"Credit spread increased by 50bp ({shock_size:.2f}); "
                                    f"FSI shift = beta_Spread ({spread_beta:.4f}) * shock."
                                )
                            elif scenario_choice == "Credit spread shock plus 100bp":
                                shock_size = 1.00  # 100 basis points expressed in spread units
                                shock_shift = spread_beta * shock_size
                                shock_desc = (
                                    f"Credit spread increased by 100bp ({shock_size:.2f}); "
                                    f"FSI shift = beta_Spread ({spread_beta:.4f}) * shock."
                                )

                            # Use ARIMA to generate baseline forecast path
                            arima_order = None
                            if models_ready and results is not None:
                                arima_order = results["arima"]["order"]
                            if arima_order is None:
                                arima_order = select_arima(sim_train["FSI"])

                            arima_res = run_arima(sim_train["FSI"], arima_order)
                            forecast_df = arima_res["forecast"](sim_horizon)

                            # Extract the baseline mean forecast
                            baseline_mean = forecast_df["mean"].copy()

                            # If forecast index is not datetime, create a daily future date index
                            if not isinstance(baseline_mean.index, pd.DatetimeIndex):
                                future_index = pd.date_range(
                                    sim_train.index.max() + pd.Timedelta(days=1), periods=sim_horizon, freq="D"
                                )
                                baseline_mean.index = future_index

                            # Shocked path is baseline plus the constant shift derived from OLS
                            shocked_mean = baseline_mean + shock_shift


                            # --- anchor forecasts to last historical point (prevents "floating") ---
                            # Add the final historical value as a point at the forecast origin,
                            # so forecast lines connect smoothly to the observed series.
                            last_date = sim_train.index[-1]
                            last_val = sim_train["FSI"].iloc[-1]

                            baseline_mean_anchored = baseline_mean.copy()
                            baseline_mean_anchored.loc[last_date] = last_val
                            baseline_mean_anchored = baseline_mean_anchored.sort_index()

                            shocked_mean_anchored = shocked_mean.copy()
                            shocked_mean_anchored.loc[last_date] = last_val
                            shocked_mean_anchored = shocked_mean_anchored.sort_index()

                            # Historical window for plotting context
                            history = sim_train["FSI"].iloc[-200:].rename("Historical FSI")

                            # Combine baseline and shocked paths into a DataFrame
                            future_paths = pd.DataFrame({
                                "Baseline forecast": baseline_mean_anchored,
                                "Shock scenario": shocked_mean_anchored
                            })

                            # Join historical and forecast paths for a single timeline plot
                            combined_paths = pd.concat([history, future_paths], axis=1)


                            # --- plot baseline vs shock scenario ---
                            fig, ax = plt.subplots(figsize=(7, 3.5))

                            # Plot historical path
                            ax.plot(combined_paths.index, combined_paths["Historical FSI"], label="Historical FSI")

                            # Plot baseline and shock paths
                            ax.plot(combined_paths.index, combined_paths["Baseline forecast"], label="Baseline forecast")
                            ax.plot(combined_paths.index, combined_paths["Shock scenario"], label="Shock scenario")

                            # Add a vertical marker where the forecast begins (forecast origin)
                            forecast_start = baseline_mean.index.min()
                            ax.axvline(forecast_start, linestyle="--", alpha=0.5)

                            # Titles and labels
                            ax.set_title("FSI: baseline vs shock scenario")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("FSI")
                            ax.legend()

                            # Render the plot in Streamlit
                            st.pyplot(fig, use_container_width=False)


                            # Display explanation of the applied shock and run settings
                            st.write(shock_desc)
                            st.write(
                                f"Simulation horizon: {sim_horizon} days. Training data through {train_end} "
                                f"| Current VIX {current_vix:.2f}, Spread {current_spread:.2f}."
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            st.error(f"Simulation failed: {exc}")


# Standard Python entry point guard
# Streamlit executes the script top-to-bottom, but this keeps it runnable as a normal script too
if __name__ == "__main__":
            main()

# comments by chatgpt edited by group ....
