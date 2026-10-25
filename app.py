from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

st.set_page_config(
    page_title="RevenueIQ | Revenue Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

LIGHT = {
    "page": "#F5F2EA",
    "surface": "#FFFFFF",
    "surface2": "#EBE8DE",
    "text": "#26352B",
    "muted": "#707A70",
    "primary": "#245640",
    "accent": "#BD7848",
    "border": "#DDDCD2",
    "grid": "#E8E6DE",
}

DARK = {
    "page": "#202723",
    "surface": "#2B342E",
    "surface2": "#354037",
    "text": "#F5F2EA",
    "muted": "#B8C2BA",
    "primary": "#A9C4AF",
    "accent": "#D7B47C",
    "border": "#465148",
    "grid": "#3B463E",
}

C = LIGHT if st.session_state.theme == "Light" else DARK

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {C["page"]};
        color: {C["text"]};
    }}
    [data-testid="stSidebar"] {{
        background-color: {C["surface"]};
        border-right: 1px solid {C["border"]};
    }}
    h1, h2, h3 {{
        color: {C["text"]} !important;
        letter-spacing: -0.035em;
    }}
    p, label, li, .stMarkdown {{
        color: {C["text"]};
    }}
    [data-testid="stMetric"] {{
        background-color: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 18px 20px;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {C["muted"]} !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricValue"] {{
        color: {C["text"]} !important;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {C["border"]};
        border-radius: 8px;
        overflow: hidden;
    }}
    div.stButton > button {{
        border-radius: 8px;
        border: 1px solid {C["border"]};
    }}
    hr {{
        border-color: {C["border"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data loading ----------
@st.cache_data
def load_csv(filename, date_columns=None):
    path = DATA / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=date_columns or [])


weekly = load_csv("weekly_revenue.csv", ["week_ending"])
anomalies = load_csv("anomaly_method_comparison.csv", ["week_ending"])
spike = load_csv("spike_investigation.csv", ["invoice_date"])
evaluation = load_csv("forecast_results.csv", ["week_ending"])
future = load_csv("future_forecast.csv", ["week_ending"])
rolling = load_csv("rolling_validation_results.csv", ["week_ending"])
forecast_comparison = load_csv("forecast_model_comparison.csv")
products_by_week = load_csv("weekly_product_contributions.csv", ["week_ending"])
invoices_by_week = load_csv("weekly_invoice_contributions.csv", ["week_ending"])
quality_by_week = load_csv("weekly_transaction_quality.csv", ["week_ending"])

if weekly is None:
    st.error("Weekly revenue data is missing.")
    st.info("Run the data cleaning and weekly aggregation scripts locally.")
    st.stop()

if anomalies is not None:
    for col in ["robust_flag", "agreement"]:
        if col in anomalies.columns:
            anomalies[col] = (
                anomalies[col].astype(str).str.strip().str.lower()
                .isin(["true", "1"])
            )


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font={"color": C["text"], "family": "Arial, sans-serif"},
        title_font={"size": 19, "color": C["text"]},
        margin={"l": 12, "r": 12, "t": 60, "b": 12},
        legend={"bgcolor": C["surface"]},
        hoverlabel={"bgcolor": C["surface"], "font_color": C["text"]},
    )
    fig.update_xaxes(
        gridcolor=C["grid"],
        linecolor=C["border"],
        zerolinecolor=C["grid"],
        tickfont={"color": C["text"]},
        title_font={"color": C["text"]},
    )
    fig.update_yaxes(
        gridcolor=C["grid"],
        linecolor=C["border"],
        zerolinecolor=C["grid"],
        tickfont={"color": C["text"]},
        title_font={"color": C["text"]},
    )
    return fig


def section_heading(kicker, heading, description=None):
    st.caption(kicker.upper())
    st.header(heading)
    if description:
        st.write(description)


# ---------- Sidebar / branding ----------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:12px 0 22px 0;">
          <div style="font-size:12px;letter-spacing:3px;
                      color:{C["primary"]};font-weight:700;">
            REVENUE / INTELLIGENCE
          </div>
          <div style="font-size:30px;font-weight:700;
                      letter-spacing:-1.5px;color:{C["text"]};">
            RevenueIQ<span style="color:{C["accent"]};">.</span>
          </div>
          <div style="font-size:12px;color:{C["muted"]};">
            Retail analytics workspace
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("APPEARANCE")
    theme_choice = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigation",
        ["Overview", "Forecasts", "Investigations", "Data Quality"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("DATASET")
    st.write("UCI Online Retail II")
    st.caption(
        f"{len(weekly):,} weekly observations · "
        f"{weekly['week_ending'].min():%b %Y} – "
        f"{weekly['week_ending'].max():%b %Y}"
    )


# ---------- Global filters ----------
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;
                align-items:center;border-bottom:1px solid {C["border"]};
                padding:4px 0 14px 0;margin-bottom:22px;">
      <span style="font-size:12px;letter-spacing:1.5px;
                   color:{C["muted"]};">REVENUE INTELLIGENCE / ANALYTICS</span>
      <span style="font-size:12px;color:{C["primary"]};">
        EXPLORATORY ANALYTICS
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

minimum = weekly["week_ending"].min().date()
maximum = weekly["week_ending"].max().date()

with st.expander("Filter historical period", expanded=False):
    date_range = st.date_input(
        "Week-ending date range",
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
    st.warning("No observations match the selected date range.")
    st.stop()

filtered_dates = set(filtered["week_ending"].dt.normalize())

shown_anomalies = (
    anomalies[
        anomalies["week_ending"].dt.normalize().isin(filtered_dates)
    ].copy()
    if anomalies is not None else pd.DataFrame()
)

if not shown_anomalies.empty and "z_score" in shown_anomalies.columns:
    shown_anomalies["standard_flag"] = shown_anomalies["z_score"].abs() >= 2

robust_flags = (
    shown_anomalies["robust_flag"]
    if "robust_flag" in shown_anomalies.columns
    else pd.Series(False, index=shown_anomalies.index)
)


# ---------- Overview ----------
if page == "Overview":
    section_heading(
        "01 / Overview",
        "Revenue at a glance.",
        "Explore historical retail value, identify unusual weeks, "
        "and understand the signals behind the numbers.",
    )

    latest = filtered.iloc[-1]
    flagged_count = int(robust_flags.sum())

    a, b, c, d = st.columns(4)
    a.metric("Weeks analyzed", f"{len(filtered):,}")
    b.metric("Total candidate value", f"{filtered['revenue'].sum():,.2f}")
    c.metric("Average weekly value", f"{filtered['revenue'].mean():,.2f}")
    d.metric("Robust flags", f"{flagged_count:,}")

    st.caption(
        "Candidate value is not verified net revenue. "
        "Currency units have not been assumed."
    )

    st.markdown("#### Revenue trend")
    fig = px.line(
        filtered,
        x="week_ending",
        y="revenue",
        markers=True,
        labels={
            "week_ending": "Week ending",
            "revenue": "Candidate value",
        },
    )
    fig.update_traces(
        line={"color": C["primary"], "width": 3},
        marker={"color": C["accent"], "size": 5},
    )

    if not shown_anomalies.empty and "robust_flag" in shown_anomalies.columns:
        points = shown_anomalies[shown_anomalies["robust_flag"]]
        if not points.empty:
            fig.add_trace(
                go.Scatter(
                    x=points["week_ending"],
                    y=points["revenue"],
                    mode="markers",
                    name="Robust anomaly",
                    marker={
                        "color": C["accent"],
                        "size": 11,
                        "symbol": "diamond",
                    },
                )
            )

    st.plotly_chart(style_chart(fig), width="stretch")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("#### Unusual weeks")
        if shown_anomalies.empty:
            st.info("No anomaly comparison data for this period.")
        else:
            flags = shown_anomalies[
                shown_anomalies.get(
                    "robust_flag", pd.Series(False, index=shown_anomalies.index)
                )
            ].copy()
            if flags.empty:
                st.success("No robust-method flags in this period.")
            else:
                display_cols = [
                    col for col in [
                        "week_ending", "revenue", "z_score", "modified_z_score"
                    ] if col in flags.columns
                ]
                st.dataframe(
                    flags[display_cols].sort_values(
                        "revenue", ascending=False
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with right:
        st.markdown("#### Reading the data")
        st.markdown(
            f"""
            **{len(filtered):,} weeks** are included in this view.

            The highest weekly candidate value in the selected period was
            **{filtered['revenue'].max():,.2f}**.

            Anomaly flags are prompts for investigation, not proof of
            fraud or data errors.
            """
        )


# ---------- Forecasts ----------
elif page == "Forecasts":
    section_heading(
        "02 / Forecasting",
        "Looking ahead.",
        "Compare forecasting models on rolling historical test windows "
        "and inspect the selected model's 12-week projection.",
    )

    st.markdown("#### Rolling-origin model comparison")

    if forecast_comparison is not None and {
        "fold", "model", "mae", "rmse"
    }.issubset(forecast_comparison.columns):

        comparison = forecast_comparison.copy()
        comparison["fold"] = pd.to_numeric(comparison["fold"], errors="coerce")
        comparison["mae"] = pd.to_numeric(comparison["mae"], errors="coerce")
        comparison["rmse"] = pd.to_numeric(comparison["rmse"], errors="coerce")
        if "test_weeks" in comparison.columns:
            comparison["test_weeks"] = pd.to_numeric(
                comparison["test_weeks"], errors="coerce"
            )

        comparison = comparison.dropna(subset=["fold", "mae", "rmse"])

        # Keep only models evaluated on every shared validation fold.
        all_folds = set(comparison["fold"].unique())
        valid_models = [
            model
            for model, frame in comparison.groupby("model")
            if set(frame["fold"].unique()) == all_folds
        ]
        comparison = comparison[comparison["model"].isin(valid_models)]

        if not comparison.empty:
            aggregation = {
                "folds": ("fold", "nunique"),
                "mean_fold_mae": ("mae", "mean"),
                "mean_fold_rmse": ("rmse", "mean"),
            }
            if "test_weeks" in comparison.columns:
                aggregation["total_test_windows"] = ("test_weeks", "sum")

            summary = (
                comparison.groupby("model", as_index=False)
                .agg(**aggregation)
                .sort_values("mean_fold_mae")
                .reset_index(drop=True)
            )

            best = summary.iloc[0]
            st.success(
                f"Best model by mean fold MAE: {best['model']} "
                f"({best['mean_fold_mae']:,.2f})."
            )

            display_columns = {
                "model": "Model",
                "folds": "Validation folds",
                "mean_fold_mae": "Mean fold MAE",
                "mean_fold_rmse": "Mean fold RMSE",
            }
            if "total_test_windows" in summary.columns:
                display_columns["total_test_windows"] = "Test predictions"

            st.dataframe(
                summary.rename(columns=display_columns),
                width="stretch",
                hide_index=True,
                column_config={
                    "Mean fold MAE": st.column_config.NumberColumn(format="%.2f"),
                    "Mean fold RMSE": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            chart = px.bar(
                summary.sort_values("mean_fold_mae", ascending=True),
                x="mean_fold_mae",
                y="model",
                orientation="h",
                labels={
                    "mean_fold_mae": "Mean fold MAE (lower is better)",
                    "model": "Model",
                },
            )
            chart.update_traces(
                marker_color="#245640"
                if st.session_state.theme == "Light"
                else "#A9C4AF"
            )
            st.plotly_chart(style_chart(chart), width="stretch")

            with st.expander("View per-fold model performance"):
                fold_display = [
                    column for column in [
                        "fold", "train_weeks", "test_weeks", "model", "mae", "rmse"
                    ] if column in comparison.columns
                ]
                st.dataframe(
                    comparison[fold_display]
                    .sort_values(["fold", "mae"])
                    .rename(columns={
                        "fold": "Fold",
                        "train_weeks": "Training weeks",
                        "test_weeks": "Test weeks",
                        "model": "Model",
                        "mae": "MAE",
                        "rmse": "RMSE",
                    }),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "MAE": st.column_config.NumberColumn(format="%.2f"),
                        "RMSE": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            st.caption(
                "Lower MAE and RMSE indicate better historical predictions. "
                "Models are compared across the same validation folds, with each fold "
                "weighted equally. Results describe this dataset and do not guarantee "
                "future accuracy."
            )
        else:
            st.warning("No comparable model results were found.")
    else:
        st.warning(
            "Model-comparison results are missing or incomplete. Run "
            "`python src/forecasting/compare_forecast_models.py` first."
        )

    st.divider()
    st.markdown("#### 12-week forecast")

    if future is not None and {
        "week_ending", "forecast_revenue"
    }.issubset(future.columns):
        st.info(
            "Selected method: seasonal naive with a 52-week lag. Each forecast "
            "repeats the corresponding observation from the previous year."
        )
        st.warning(
            "These are estimates, not guaranteed outcomes. The method assumes "
            "the annual pattern persists and does not include prediction intervals."
        )

        future_fig = px.line(
            future,
            x="week_ending",
            y="forecast_revenue",
            markers=True,
            labels={
                "week_ending": "Week ending",
                "forecast_revenue": "Forecast candidate value",
            },
        )
        future_fig.update_traces(
            line={"color": C["accent"], "width": 3},
            marker={"color": C["primary"], "size": 6},
        )
        st.plotly_chart(style_chart(future_fig), width="stretch")
        st.dataframe(future, width="stretch", hide_index=True)
    else:
        st.info("Future forecast data is missing.")


# ---------- Investigations ----------
elif page == "Investigations":
    section_heading(
        "03 / Investigations",
        "Find what moved the numbers.",
        "Trace unusual weekly revenue to products, invoices and transaction signals.",
    )

    if products_by_week is None or invoices_by_week is None or quality_by_week is None:
        st.warning("Investigation summary files are missing.")
        st.info(
            "Run `python src/data/build_transaction_investigation.py` "
            "to generate the product, invoice and weekly quality summaries."
        )
    else:
        product_data = products_by_week.copy()
        invoice_data = invoices_by_week.copy()
        quality_data = quality_by_week.copy()

        for frame in [product_data, invoice_data, quality_data]:
            frame["week_ending"] = pd.to_datetime(
                frame["week_ending"], errors="coerce"
            ).dt.normalize()

        available_weeks = sorted(
            set(product_data["week_ending"].dropna())
            & set(invoice_data["week_ending"].dropna())
            & set(quality_data["week_ending"].dropna())
        )

        if not available_weeks:
            st.warning("No matching investigation weeks were found.")
        else:
            flagged = (
                shown_anomalies[
                    shown_anomalies.get(
                        "robust_flag",
                        pd.Series(False, index=shown_anomalies.index),
                    )
                ]["week_ending"].dt.normalize().tolist()
                if not shown_anomalies.empty
                and "robust_flag" in shown_anomalies.columns
                else []
            )
            eligible_weeks = [week for week in available_weeks if week in set(filtered_dates)]
            flagged_weeks = [week for week in eligible_weeks if week in set(flagged)]

            st.markdown("#### Choose a week to investigate")
            if flagged_weeks:
                default_week = flagged_weeks[-1]
                week_options = flagged_weeks
                st.caption(
                    "Showing robust-anomaly weeks inside the selected historical period. "
                    "You can switch to all weeks below."
                )
                investigate_all = st.checkbox(
                    "Show all weeks in the selected historical period",
                    value=False,
                )
                if investigate_all:
                    week_options = eligible_weeks
            else:
                week_options = eligible_weeks
                st.caption(
                    "No robust anomaly is flagged in the selected historical period. "
                    "You can still investigate any available week."
                )

            if not week_options:
                st.info("No investigation weeks fall inside the selected date range.")
            else:
                if "default_week" not in locals() or default_week not in week_options:
                    default_week = week_options[-1]
                selected_week = st.selectbox(
                    "Week ending",
                    options=week_options,
                    index=week_options.index(default_week),
                    format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
                )

                week_products = product_data[
                    product_data["week_ending"].eq(selected_week)
                ].copy()
                week_invoices = invoice_data[
                    invoice_data["week_ending"].eq(selected_week)
                ].copy()
                week_quality = quality_data[
                    quality_data["week_ending"].eq(selected_week)
                ].copy()

                weekly_row = filtered[
                    filtered["week_ending"].dt.normalize().eq(selected_week)
                ]
                weekly_revenue = (
                    float(weekly_row.iloc[0]["revenue"])
                    if not weekly_row.empty
                    else float(week_quality["candidate_revenue"].iloc[0])
                )

                previous_week = selected_week - pd.Timedelta(days=7)
                previous_products = product_data[
                    product_data["week_ending"].eq(previous_week)
                ][["stockcode", "description", "revenue"]].rename(
                    columns={"revenue": "previous_week_revenue"}
                )

                week_products = week_products.merge(
                    previous_products,
                    on=["stockcode", "description"],
                    how="left",
                )
                week_products["previous_week_revenue"] = (
                    week_products["previous_week_revenue"].fillna(0)
                )
                week_products["revenue_change"] = (
                    week_products["revenue"]
                    - week_products["previous_week_revenue"]
                )
                week_products["share_of_week_pct"] = (
                    week_products["revenue"] / weekly_revenue * 100
                    if weekly_revenue
                    else 0.0
                )

                q = week_quality.iloc[0]
                a, b, c, d = st.columns(4)
                a.metric("Candidate value", f"{weekly_revenue:,.2f}")
                b.metric("Invoices", f"{int(q['invoice_count']):,}")
                c.metric("Products", f"{int(q['product_count']):,}")
                d.metric("Transaction lines", f"{int(q['transaction_lines']):,}")

                st.caption(
                    "Values are candidate sales value, not reconciled net revenue. "
                    "Large quantities and line values are signals for review, not "
                    "automatic evidence of errors."
                )

                st.markdown("#### Products contributing most")
                top_products = week_products.sort_values(
                    "revenue", ascending=False
                ).head(15)
                product_chart = px.bar(
                    top_products.sort_values("revenue"),
                    x="revenue",
                    y="description",
                    orientation="h",
                    labels={
                        "revenue": "Candidate value",
                        "description": "Product",
                    },
                )
                product_chart.update_traces(marker_color=C["primary"])
                st.plotly_chart(style_chart(product_chart), width="stretch")

                product_cols = [
                    col for col in [
                        "stockcode", "description", "revenue",
                        "share_of_week_pct", "total_quantity", "invoice_count",
                        "high_quantity_lines", "high_value_lines",
                    ] if col in week_products.columns
                ]
                st.dataframe(
                    top_products[product_cols],
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Product changes versus the previous week")
                changes = week_products.sort_values(
                    "revenue_change", ascending=False
                )
                left, right = st.columns(2)
                with left:
                    st.markdown("**Largest increases**")
                    st.dataframe(
                        changes.head(10)[
                            ["stockcode", "description", "previous_week_revenue",
                             "revenue", "revenue_change"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                with right:
                    st.markdown("**Largest decreases**")
                    st.dataframe(
                        changes.tail(10).sort_values("revenue_change")[
                            ["stockcode", "description", "previous_week_revenue",
                             "revenue", "revenue_change"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )

                st.markdown("#### Largest invoices")
                invoice_cols = [
                    col for col in [
                        "invoice", "revenue", "total_quantity", "line_count",
                        "product_count", "high_quantity_lines", "high_value_lines",
                    ] if col in week_invoices.columns
                ]
                st.dataframe(
                    week_invoices.sort_values("revenue", ascending=False)
                    .head(20)[invoice_cols],
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Transaction signals")
                signal_a, signal_b, signal_c, signal_d = st.columns(4)
                signal_a.metric(
                    "High-quantity lines",
                    f"{int(q['high_quantity_lines']):,}",
                )
                signal_b.metric(
                    "High-value lines",
                    f"{int(q['high_value_lines']):,}",
                )
                signal_c.metric("Largest quantity", f"{q['largest_quantity']:,.0f}")
                signal_d.metric("Largest line value", f"{q['largest_line_value']:,.2f}")

                with st.expander("View all product records for this week"):
                    st.dataframe(
                        week_products.sort_values("revenue", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                with st.expander("View all invoice records for this week"):
                    st.dataframe(
                        week_invoices.sort_values("revenue", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )

    st.caption(
        "Anomaly detection identifies unusual observations. It does not "
        "establish fraud, invalid transactions, or business causes."
    )


# ---------- Data quality ----------
elif page == "Data Quality":
    section_heading(
        "04 / Data quality",
        "Know the data behind the numbers.",
        "Review the source dataset, cleaning decisions and analytical limitations.",
    )

    profile_path = ROOT / "data_profile.md"
    if profile_path.exists():
        st.markdown("#### Dataset profile")
        st.markdown(profile_path.read_text(encoding="utf-8"))

    st.markdown("#### Weekly series checks")
    a, b, c, d = st.columns(4)
    a.metric("Weekly observations", f"{len(weekly):,}")
    b.metric("Missing revenue values", f"{weekly['revenue'].isna().sum():,}")
    bdupes = weekly["week_ending"].duplicated().sum()
    c.metric("Duplicate week dates", f"{bdupes:,}")
    d.metric("Zero-value weeks", f"{(weekly['revenue'] == 0).sum():,}")

    st.markdown("#### Cleaning and interpretation notes")
    st.markdown(
        """
        - Exact duplicate rows were removed, keeping the first copy.
        - Positive quantity and price rows are used to create candidate sales.
        - Cancellation invoices and returns require reconciliation before
          candidate values can be treated as net revenue.
        - Missing customer IDs are retained in the cleaned data.
        - Large transaction quantities may be legitimate bulk purchases and
          should be investigated rather than automatically discarded.
        """
    )

    st.markdown("#### Investigation coverage")
    if (
        products_by_week is not None
        and invoices_by_week is not None
        and quality_by_week is not None
    ):
        qa, qb, qc = st.columns(3)
        qa.metric("Weekly product records", f"{len(products_by_week):,}")
        qb.metric("Weekly invoice records", f"{len(invoices_by_week):,}")
        qc.metric("Weekly quality summaries", f"{len(quality_by_week):,}")
        st.caption(
            "Investigation summaries cover all available sale-candidate weeks. "
            "High-quantity and high-value thresholds are review signals, not error labels."
        )
    else:
        st.warning(
            "Investigation summaries are not available. Run "
            "`python src/data/build_transaction_investigation.py`."
        )

    with st.expander("Technical limitations"):
        st.markdown(
            """
            - The historical dataset contains 106 weekly observations.
            - Rolling validation covers four non-overlapping 12-week test windows.
            - Forecast performance may not generalize to other retail datasets.
            - The seasonal-naive future projection repeats values from 52 weeks earlier and has no prediction intervals.
            - Anomaly flags are exploratory signals, not confirmed errors or fraud.
            """
        )


# ---------- Footer ----------
st.divider()
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;gap:12px;
                flex-wrap:wrap;color:{C["muted"]};font-size:12px;">
      <span>REVENUEIQ · REVENUE INTELLIGENCE</span>
      <span>Exploratory analytics prototype · UCI Online Retail II</span>
    </div>
    """,
    unsafe_allow_html=True,
)