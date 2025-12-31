# ---
# description: Provides a Streamlit app to visualize stock forecast metrics and actual prices.
# ---

# ---

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# from pandas import Timestamp
import signal_sigma.config.cfg as cfg
import streamlit as st

# === Page config ===
st.set_page_config(layout="wide", page_title="Signal Sigma - Stock Forecasting")

# === Configuration ===
start_date = "2014-01-01"
end_date = "2025-05-17"
path_stock = os.path.join(cfg.DATA_PATH, "stock_market_data")

# === Metric Explanations ===
metric_explanations = {
    "mae_point": {
        "label": "MAE (Point Forecast)",
        "description": "Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual values.\n\n**Formula:**  \nMAE = (1/n) * Σ |yᵢ - ŷᵢ|\n\n**Interpretation:**  \nLower is better. MAE gives an average magnitude of errors without considering their direction.",
        "friendly": "This shows how far off the forecast was on average. It tells you how wrong the model is, in dollars. Smaller is better.",
    },
    "r2_point": {
        "label": "R² (Point Forecast)",
        "description": "R² (Coefficient of Determination) explains how well the forecast captures variance in the actual data.\n\n**Formula:**  \nR² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)\n\n**Interpretation:**  \nRanges from -∞ to 1. Closer to 1 is better. Negative means the model performs worse than predicting the mean.",
        "friendly": "This tells you how much of the ups and downs in the stock the model could explain. If it's close to 1, the model did well.",
    },
    "mae_p50": {
        "label": "MAE (p50 Forecast)",
        "description": "MAE applied to the p50 (median) quantile forecast. Same interpretation as standard MAE, applied to the 50th percentile prediction.",
        "friendly": "Same as MAE, but based on the 50th percentile (the model's 'most typical' guess).",
    },
    "r2_p50": {
        "label": "R² (p50 Forecast)",
        "description": "R² for the p50 quantile forecast. Same concept as point forecast R², but for the median forecast values.",
        "friendly": "Same as R², but using the median forecast instead of a single-point prediction.",
    },
    "pinball_p10": {
        "label": "Pinball Loss (p10)",
        "description": "Pinball Loss evaluates the accuracy of quantile forecasts.\n\n**Formula (for quantile q):**  \nLoss = max(q(y - ŷ), (q - 1)(y - ŷ))\n\n**Interpretation:**  \nLower is better. Measures how well the predicted quantile (e.g., 10th percentile) captures the uncertainty.",
        "friendly": "This tells you how well the model predicted 'low-case' scenarios (worst 10%). Lower is better.",
    },
    "pinball_p50": {
        "label": "Pinball Loss (p50)",
        "description": "Pinball Loss for the 50th percentile forecast. Equivalent to MAE when q = 0.5.",
        "friendly": "Same as MAE but for the median prediction. Useful when modeling uncertainty.",
    },
    "pinball_p90": {
        "label": "Pinball Loss (p90)",
        "description": "Pinball Loss for the 90th percentile. Helps assess the accuracy of upper-bound forecasts.",
        "friendly": "Shows how well the model predicted 'high-case' scenarios (best 10%). Lower = better upper-bound estimates.",
    },
}

# === Sidebar: Stock and Plot Type Selection ===
st.sidebar.header("⚙️ Configuration")

stock_options = {
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Apple (AAPL)": "AAPL",
    "Meta (META)": "META",
}
selected_display = st.sidebar.selectbox(
    "Choose a stock:", list(stock_options.keys()), index=1
)
stock = stock_options[selected_display]
plot_mode = st.sidebar.radio(
    "Select Plot Type",
    ["Forecast", "Daily Metric", "Summary Metrics", "All Stocks Actual"],
)
# plot_mode = st.sidebar.radio("Select Plot Type", ["Forecast", "Daily Metric", "Summary Metrics"])


# === Load Data ===

# XXX: Ugly hack in order to not guess the date range
forecast_filenames_guessed = [
    f for f in os.listdir(path_stock) if f.startswith(f"{stock}_forecast_eval_metrics_")
]

try:
    forecast_filename = forecast_filenames_guessed[0]
except IndexError:
    forecast_filename = ""

raw_filenames_guessed = [
    f for f in os.listdir(path_stock) if f.startswith(f"{stock}_reduced_dataset_")
]

try:
    raw_filename = raw_filenames_guessed[0]
except IndexError:
    raw_filename = ""

# # XXX: Please uncomment the following lines when you have a better way to guess the date range
# forecast_filename = f"{stock}_forecast_eval_metrics_{start_date}_{end_date}.csv"
# raw_filename = f"{stock}_reduced_dataset_{start_date}_{end_date}.csv"

forecast_path = os.path.join(path_stock, forecast_filename)
raw_path = os.path.join(path_stock, raw_filename)

if not (os.path.exists(forecast_path) and os.path.exists(raw_path)):
    st.error(
        f"Data files for {stock} not found. Please check the directory and file naming."
    )
    st.stop()

raw_df = pd.read_csv(
    raw_path,
    parse_dates=["date"],
    index_col="date",
).sort_index()

forecast_df = pd.read_csv(
    forecast_path, parse_dates=["date"], index_col="date"
).sort_index()

forecast_df_index = forecast_df.index
y_p10 = forecast_df["p10"]
y_p50 = forecast_df["p50"]
y_p90 = forecast_df["p90"]
y_point = forecast_df["point_forecast"]
y_actual = raw_df["target"]
test_start = forecast_df_index[0]

# === Metric List ===
daily_metrics = list(metric_explanations.keys())

# === Summary Plot ===
avg_metrics = forecast_df[daily_metrics].mean().sort_values()
summary_fig = go.Figure()
summary_fig.add_trace(
    go.Bar(x=avg_metrics.index, y=avg_metrics.values, marker=dict(color="teal"))
)
summary_fig.update_layout(
    title="Average Forecast Metrics Over Forecast Window",
    xaxis_title="Metric",
    yaxis_title="Average Value",
    xaxis_tickangle=-45,
    template="plotly_white",
    height=500,
)

# === Forecast Plot ===
if plot_mode == "Forecast":
    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=raw_df.index,
            y=y_actual,
            name="Actual Price",
            line=dict(color="orange", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df_index,
            y=y_p50,
            name="p50 Quantile Forecast",
            line=dict(color="magenta", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df_index,
            y=y_point,
            name="Single Point Forecast",
            line=dict(color="blue", dash="dot", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df_index,
            y=y_p90,
            name="p90 Quantile Forecast",
            line=dict(color="green", width=1),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df_index,
            y=y_p10,
            name="p10 Quantile Forecast",
            line=dict(color="red", width=1),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df_index,
            y=y_p10,
            name="p10–p90 Confidence Interval",
            fill="tonexty",
            fillcolor="rgba(173,216,230,0.3)",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )
    forecast_fig.add_shape(
        type="line",
        x0=test_start,
        x1=test_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", dash="dash", width=1),
    )
    forecast_fig.add_annotation(
        x=test_start,
        y=1.02,
        xref="x",
        yref="paper",
        showarrow=False,
        text="Forecast Start",
        font=dict(color="red", size=11),
    )
    forecast_fig.update_layout(
        title=f"{stock} Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="Price (US$)",
        hovermode="x unified",
        template="plotly_white",
        height=750,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )

# === All Stocks Actual Plot ===

elif plot_mode == "All Stocks Actual":
    st.markdown("### 📊 All Stocks: Actual Price Trends")
    stock_data = {}
    for label, symbol in stock_options.items():
        # XXX: Ugly hack in order to not guess the date range
        filenames_guessed = [
            f
            for f in os.listdir(path_stock)
            if f.startswith(f"{symbol}_reduced_dataset_")
        ]

        try:
            filename = filenames_guessed[0]
        except IndexError:
            filename = ""
        path = os.path.join(path_stock, filename)

        # # XXX: Please uncomment the following lines when you have a better way to guess the date range
        # path = f"{path_stock}/{symbol}_reduced_dataset_{start_date}_{end_date}.csv"

        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df["Stock"] = label
            df = df[["date", "target", "Stock"]].rename(
                columns={"target": "Actual Price"}
            )
            stock_data[label] = df

    if stock_data:
        all_data = pd.concat(stock_data.values(), ignore_index=True)
        fig_all = px.line(
            all_data,
            x="date",
            y="Actual Price",
            color="Stock",
            title="Actual Stock Prices Across All Companies",
        )
        # st.plotly_chart(fig_all, use_container_width=True)
        st.plotly_chart(fig_all, use_container_width=True, key="all_stocks_chart")

    else:
        st.warning("No valid stock data found for any of the companies.")


# === Layout ===
st.title("📈 Stock Forecast Visualization")
left_col, right_col = st.columns([1, 3])
if plot_mode == "Forecast":
    st.markdown(
        """
    This dashboard visualizes forecasted stock prices using two prediction approaches:

    - **Quantile Forecasting (p10, p50, p90)**: These are estimates of a range of possible outcomes.
      - p10 = conservative/lower estimate
      - p50 = typical/median estimate
      - p90 = optimistic/upper estimate
      - The shaded area between p10 and p90 shows the uncertainty interval. Narrower is more confident.

    - **Single Point Forecast**: A single best-guess prediction.

    ### How to interpret:
    - If actual price stays within the p10–p90 range → the model captures uncertainty well.
    - If actual price tracks close to p50 → the median forecast is well-calibrated.
    - If point forecast tracks better than p50 → point modeling is better suited.

    Use this chart to understand future confidence levels, risk, and where predictions may deviate.
    """
    )


with left_col:
    st.markdown("### ℹ️ Details")
    st.markdown(f"**Selected Stock:** {stock}")
    st.markdown(f"**Date Range:** {start_date} → {end_date}")

with right_col:
    if plot_mode == "Forecast":
        st.markdown("### Forecast Plot")
        # st.plotly_chart(forecast_fig, use_container_width=True)
        st.plotly_chart(
            forecast_fig, use_container_width=True, key="forecast_chart_main"
        )

    elif plot_mode == "Daily Metric":
        st.markdown("### Daily Metric")

        selected_metric = st.selectbox(
            "Select Daily Metric", daily_metrics, key="daily_metric_dropdown"
        )
        explanation = metric_explanations.get(selected_metric, {})

        show_explanation = st.checkbox("Show explanation", value=False)

        if explanation and show_explanation:
            st.markdown(f"**🧠 {explanation['label']}**")
            st.info(f"**Technical Explanation**\n\n{explanation['description']}")
            st.success(f"**Plain Summary**\n\n{explanation['friendly']}")

        metric_fig = go.Figure()
        metric_fig.add_trace(
            go.Scatter(
                x=forecast_df_index,
                y=forecast_df[selected_metric],
                mode="lines+markers",
                name=selected_metric,
                line=dict(color="orange", width=2),
            )
        )
        metric_fig.update_layout(
            title=f"Daily {selected_metric} for {stock}",
            xaxis_title="Date",
            yaxis_title=selected_metric,
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(metric_fig, use_container_width=True)

    elif plot_mode == "Summary Metrics":
        st.markdown("### Summary Metrics")
        st.plotly_chart(summary_fig, use_container_width=True)

        st.markdown("### 🧠 Metric Interpretation Summary")
        for key in avg_metrics.index:
            exp = metric_explanations.get(key, {})
            if exp:
                st.markdown(f"- **{exp['label']}**: {exp['friendly']}")
