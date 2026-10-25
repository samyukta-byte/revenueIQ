
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

st.set_page_config(
    page_title="RevenueIQ Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("RevenueIQ Intelligence")
st.caption("Explainable revenue analytics | Forecasting | Anomaly investigation")


@st.cache_data
def load_data(filename, dates=None):
    path = DATA / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=dates or [])


weekly = load_data("weekly_revenue.csv", ["week_ending"])
anomalies = load_data("anomaly_method_comparison.csv", ["week_ending"])
spike = load_data("spike_investigation.csv", ["invoice_date"])
evaluation = load_data("forecast_results.csv", ["week_ending"])
future = load_data("future_forecast.csv", ["week_ending"])
rolling = load_data("rolling_validation_results.csv", ["week_ending"])

if weekly is None:
    st.error("weekly_revenue.csv is missing. Run your data pipeline first.")
    st.stop()

# Normalize anomaly flags
if anomalies is not None:
    for col in ["robust_flag", "agreement"]:
        if col in anomalies.columns:
            anomalies[col] = (
                anomalies[col].astype(str).str.lower().str.strip()
                .isin(["true", "1"])
            )

# Sidebar filters
st.sidebar.header("Filters")
minimum = weekly["week_ending"].min().date()
maximum = weekly["week_ending"].max().date()

date_range = st.sidebar.date_input(
    "Historical date range",
    value=(minimum, maximum),
    min_value=minimum,
    max_value=maximum,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    filtered = weekly[
        weekly["week_ending"].dt.date.between(start, end)
    ].copy()
else:
    filtered = weekly.copy()

if filtered.empty:
    st.warning("No data available for this date range.")
    st.stop()

# Overview
st.header("1. Revenue overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Weeks analyzed", f"{len(filtered):,}")
c2.metric("Total candidate value", f"{filtered['revenue'].sum():,.2f}")
c3.metric("Average weekly value", f"{filtered['revenue'].mean():,.2f}")
c4.metric("Peak week value", f"{filtered['revenue'].max():,.2f}")

st.caption(
    "Values are positive-sale candidate value, not verified net revenue. "
    "No currency is assumed."
)

fig = px.line(
    filtered,
    x="week_ending",
    y="revenue",
    markers=True,
    title="Weekly positive-sale candidate value",
    labels={"week_ending": "Week ending", "revenue": "Candidate value"},
)
st.plotly_chart(fig, width="stretch")

# Anomaly detection
st.header("2. Anomaly detection")

if anomalies is None:
    st.warning("anomaly_method_comparison.csv is missing.")
else:
    shown = anomalies[
        anomalies["week_ending"].isin(filtered["week_ending"])
    ].copy()

    if "z_score" not in shown.columns:
        st.warning("The anomaly file is missing its z_score column.")
    else:
        shown["standard_flag"] = shown["z_score"].abs() >= 2
        robust_count = (
            shown["robust_flag"].sum()
            if "robust_flag" in shown.columns else 0
        )
        standard_count = shown["standard_flag"].sum()

        a, b, c = st.columns(3)
        a.metric("Standard z-score flags", int(standard_count))
        b.metric("Robust-method flags", int(robust_count))
        c.metric(
            "Flagged by both",
            int(shown["agreement"].sum())
            if "agreement" in shown.columns else 0,
        )

        flagged = shown[
            shown["standard_flag"]
            | (
                shown["robust_flag"]
                if "robust_flag" in shown.columns
                else False
            )
        ]

        st.dataframe(
            flagged.sort_values("week_ending", ascending=False),
            width="stretch",
            hide_index=True,
        )

        anomaly_chart = px.scatter(
            filtered,
            x="week_ending",
            y="revenue",
            title="Historical revenue with robust anomaly flags",
            labels={"week_ending": "Week ending", "revenue": "Candidate value"},
        )

        if "robust_flag" in shown.columns:
            flagged_points = shown[shown["robust_flag"]]
            if not flagged_points.empty:
                anomaly_chart.add_scatter(
                    x=flagged_points["week_ending"],
                    y=flagged_points["revenue"],
                    mode="markers",
                    name="Robust anomaly",
                    marker={"size": 12, "symbol": "x"},
                )

        st.plotly_chart(anomaly_chart, width="stretch")

st.caption(
    "An anomaly is a signal for investigation, not proof of fraud or a data error."
)

# Transaction investigation
st.header("3. Transaction-level investigation")

if spike is None:
    st.info("Transaction investigation data is not available.")
else:
    if "invoice_date" in spike.columns:
        selected_date = st.selectbox(
            "Choose investigation week",
            options=filtered["week_ending"].tolist(),
            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
        )
        selected_date = pd.Timestamp(selected_date)
        start_date = selected_date - pd.Timedelta(days=6)
        end_date = selected_date + pd.Timedelta(days=1)

        lines = spike[
            spike["invoice_date"].ge(start_date)
            & spike["invoice_date"].lt(end_date)
        ].copy()

        if lines.empty:
            st.info(
                "Detailed transaction data is currently available only for "
                "the saved spike-investigation week."
            )
        else:
            lines["line_value"] = lines["quantity"] * lines["price"]

            a, b, c = st.columns(3)
            a.metric("Transaction lines", f"{len(lines):,}")
            b.metric("Invoices", f"{lines['invoice'].nunique():,}")
            c.metric("Products", f"{lines['stockcode'].nunique():,}")

            st.subheader("Top products by candidate value")
            if {"stockcode", "description"}.issubset(lines.columns):
                products = (
                    lines.groupby(["stockcode", "description"], dropna=False)
                    .agg(
                        quantity=("quantity", "sum"),
                        candidate_value=("line_value", "sum"),
                    )
                    .sort_values("candidate_value", ascending=False)
                    .head(10)
                    .reset_index()
                )
                st.dataframe(products, width="stretch", hide_index=True)

            st.subheader("Largest invoices")
            invoices = (
                lines.groupby("invoice", dropna=False)
                .agg(
                    candidate_value=("line_value", "sum"),
                    transaction_lines=("line_value", "size"),
                )
                .sort_values("candidate_value", ascending=False)
                .head(10)
                .reset_index()
            )
            st.dataframe(invoices, width="stretch", hide_index=True)

            with st.expander("View transaction details"):
                st.dataframe(
                    lines.sort_values("line_value", ascending=False),
                    width="stretch",
                    hide_index=True,
                )

# Original holdout evaluation
st.header("4. Forecast model evaluation — 12-week holdout")

if evaluation is None:
    st.info("Run src/forecasting/forecast_revenue.py to generate forecast results.")
else:
    required_cols = {
        "actual", "baseline_prediction", "smoothing_prediction", "week_ending"
    }

    if required_cols.issubset(evaluation.columns):
        def metrics(actual, predicted):
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            return mae, rmse

        base_mae, base_rmse = metrics(
            evaluation["actual"], evaluation["baseline_prediction"]
        )
        smooth_mae, smooth_rmse = metrics(
            evaluation["actual"], evaluation["smoothing_prediction"]
        )

        a, b = st.columns(2)
        with a:
            st.subheader("Training-mean baseline")
            st.metric("MAE", f"{base_mae:,.2f}")
            st.metric("RMSE", f"{base_rmse:,.2f}")
        with b:
            st.subheader("Exponential smoothing")
            st.metric("MAE", f"{smooth_mae:,.2f}")
            st.metric("RMSE", f"{smooth_rmse:,.2f}")

        holdout_chart = px.line(
            evaluation,
            x="week_ending",
            y=["actual", "baseline_prediction", "smoothing_prediction"],
            markers=True,
            title="Actual vs predicted values",
            labels={
                "week_ending": "Week ending",
                "value": "Candidate value",
                "variable": "Series",
            },
        )
        st.plotly_chart(holdout_chart, width="stretch")
        st.dataframe(evaluation, width="stretch", hide_index=True)

# Rolling validation
st.header("5. Rolling-origin validation")

if rolling is None:
    st.info(
        "Rolling validation results are missing. Run "
        "`python src/forecasting/forecast_revenue.py`."
    )
else:
    needed = {
        "fold", "actual", "baseline_prediction", "smoothing_prediction"
    }

    if needed.issubset(rolling.columns):
        rolling["fold"] = rolling["fold"].astype(int)

        def summarize_model(frame, prediction_col):
            actual = frame["actual"].to_numpy()
            predicted = frame[prediction_col].to_numpy()
            return {
                "MAE": np.mean(np.abs(actual - predicted)),
                "RMSE": np.sqrt(np.mean((actual - predicted) ** 2)),
            }

        base = summarize_model(rolling, "baseline_prediction")
        smooth = summarize_model(rolling, "smoothing_prediction")

        improvement = (
            (base["MAE"] - smooth["MAE"]) / base["MAE"] * 100
            if base["MAE"] else 0
        )

        a, b, c = st.columns(3)
        a.metric("Validation folds", int(rolling["fold"].nunique()))
        b.metric("Test predictions", f"{len(rolling):,}")
        c.metric("MAE improvement", f"{improvement:.2f}%")

        score_table = pd.DataFrame([
            {"Model": "Training-mean baseline", **base},
            {"Model": "Exponential smoothing", **smooth},
        ])
        st.subheader("Performance across all test windows")
        st.dataframe(score_table, width="stretch", hide_index=True)

        fold_results = []
        for fold_number, frame in rolling.groupby("fold"):
            baseline_metrics = summarize_model(frame, "baseline_prediction")
            smoothing_metrics = summarize_model(frame, "smoothing_prediction")
            fold_results.append({
                "Fold": fold_number,
                "Test weeks": len(frame),
                "Baseline MAE": baseline_metrics["MAE"],
                "Smoothing MAE": smoothing_metrics["MAE"],
                "Baseline RMSE": baseline_metrics["RMSE"],
                "Smoothing RMSE": smoothing_metrics["RMSE"],
            })

        fold_table = pd.DataFrame(fold_results)
        st.subheader("Performance by fold")
        st.dataframe(fold_table, width="stretch", hide_index=True)

        fold_chart = px.line(
            fold_table.melt(
                id_vars=["Fold"],
                value_vars=["Baseline MAE", "Smoothing MAE"],
                var_name="Model",
                value_name="MAE",
            ),
            x="Fold",
            y="MAE",
            color="Model",
            markers=True,
            title="MAE by validation fold",
        )
        st.plotly_chart(fold_chart, width="stretch")

        st.caption(
            "Rolling-origin results are more informative than one holdout, "
            "but the dataset is small and contains only 106 weekly observations. "
            "The validation windows are chronological and non-overlapping."
        )

# Future forecast
st.header("6. Future weekly forecast")

if future is None:
    st.info(
        "Run `python src/forecasting/generate_future_forecast.py` "
        "to generate future forecasts."
    )
else:
    if {"week_ending", "forecast_revenue"}.issubset(future.columns):
        st.warning(
            "These are model estimates, not guaranteed results. "
            "The current model does not include seasonality or prediction intervals."
        )

        future_chart = px.line(
            future,
            x="week_ending",
            y="forecast_revenue",
            markers=True,
            title="12-week forecast",
            labels={
                "week_ending": "Forecast week ending",
                "forecast_revenue": "Forecast candidate value",
            },
        )
        st.plotly_chart(future_chart, width="stretch")
        st.dataframe(future, width="stretch", hide_index=True)

# Limitations
with st.expander("Data definitions and limitations"):
    st.markdown(
        """
        - The source is the UCI Online Retail II dataset.
        - Positive-sale candidate value is not verified net revenue.
        - Returns, cancellations, duplicates, and missing customer IDs affect
          interpretation of the data.
        - Anomaly flags identify unusual weeks; they do not establish causes.
        - Forecast evaluation is based on historical chronological splits.
        - Rolling validation covers four test windows and 48 predictions.
        - The future forecast is a model projection, not a verified outcome.
        - Transaction drill-down currently uses a saved investigation subset.
        """
    )

st.caption("RevenueIQ Intelligence | Exploratory analytics portfolio project")
