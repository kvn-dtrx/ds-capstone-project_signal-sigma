# ---
# description: Entry point for the execution of the core module.
# ---

# NOTE: Actually, the functionally contained in this files belongs to `__main__.py`

# ---

# -------------------------------------------------------------------------------------
# 📊 TFT Time Series Forecasting Pipeline Configuration for a Target Stock
# -------------------------------------------------------------------------------------

# ===============================
# 📁 Custom Project Modules
# ===============================

import signal_sigma.config.cfg as cfg

# # Fetch stock & Yahoo macro data
# from signal_sigma.data_gathering import DataGathering

# # Prepare dataset for modeling
# from signal_sigma.data_preparator import DataPreparator

# # Generate technical indicators
# from signal_sigma.feature_engineering import (
#     FeatureEngineering,
# )

# # Download & compress FRED macroeconomic indicators
# from signal_sigma.fred_macro import (
#     FredMacroProcessor,
# )

# # Compress Yahoo macroeconomic signals
# from signal_sigma.market_macro_compressor import (
#     MarketMacroCompressor,
# )

# Main Class
from signal_sigma.core.data_engineering_pipeline import DataEngineeringPipeline

# Loss function for the model
from signal_sigma.core.loss_history import LossHistory

# ===============================
# 🧪 Core Python Libraries
# ===============================

import os
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import Timestamp

# import sys
# import copy
# import seaborn as sns

# ===============================
# ⏳ Time Series Libraries (Darts)
# ===============================

# Main time series object
from darts import TimeSeries

# Temporal Fusion Transformer model
from darts.models import TFTModel

# For quantile loss functions
from darts.utils.likelihood_models import (
    QuantileRegression,
)

# For scaling input data
from darts.dataprocessing.transformers import Scaler

# Evaluation metrics
from darts.metrics import (
    # mae,
    # rmse,
    smape,
    mape,
    r2_score,
    # mse,
    # rmsle,
)

# ===============================
# ⚙️ Machine Learning (Auxiliary)
# ===============================

# PyTorch Lightning for model training
from pytorch_lightning.callbacks.early_stopping import (
    # Early stopping callback
    EarlyStopping,
)

# Sklearn metrics
from sklearn.metrics import (
    # mean_absolute_error,
    # mean_squared_error,
    r2_score,
)

# ===============================
# 📋 Utility Tools
# ===============================

# Nice tabular printing for reporting
from tabulate import tabulate

# ===============================
# ❗ Clean Up Output
# ===============================

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# ===============================
# 🧭 Process Configuration
# ===============================

# Defaults for the pipeline.
DEFAULTS = {
    # Ticker of the target stock to forecast.
    "target_stock": "NVDA",
    # Path where the stock market data is stored.
    "path_stock": os.path.join(cfg.DATA_PATH, "stock_market_data"),
    # Start for the historical data.
    "start_date": "2014-01-01",
    # End for the historical data.
    "end_date": "2025-05-09",
    # Forecast horizon in days.
    "output_len": 15,
    # Whether to run the pipeline.
    "run_pipeline": False,
}

# Disable actual showing of plots
plt.show = lambda: None

parser = argparse.ArgumentParser(
    description="Stock Forecasting App",
)

parser.add_argument(
    "-t",
    "--target_stock",
    type=str,
    default=DEFAULTS["target_stock"],
    help=f"Target stock symbol (default: {DEFAULTS['target_stock']})",
)

parser.add_argument(
    "--path_stock",
    type=str,
    default=DEFAULTS["path_stock"],
    help=f"Path to stock data (default: {DEFAULTS['path_stock']})",
)

parser.add_argument(
    "-a",
    "--start_date",
    type=str,
    default=DEFAULTS["start_date"],
    help=f"Start date (default: {DEFAULTS['start_date']})",
)

parser.add_argument(
    "-z",
    "--end_date",
    type=str,
    default=DEFAULTS["end_date"],
    help=f"End date (default: {DEFAULTS['end_date']})",
)

parser.add_argument(
    "-r",
    "--run_pipeline",
    action="store_true",
    help="Run the forecasting pipeline (default: False)",
)

parser.add_argument(
    "-l",
    "--output_len",
    type=int,
    default=DEFAULTS["output_len"],
    help=f"Forecast horizon in days (default: {DEFAULTS['output_len']})",
)

args = parser.parse_args()

target_stock = args.target_stock
path_stock = args.path_stock
start_date = args.start_date
end_date = args.end_date
output_len = args.output_len
run_pipeline = args.run_pipeline

print("[INFO] Configuration:")
print(f"   • Target stock:   {target_stock}")
print(f"   • Data path:      {path_stock}")
print(f"   • Start date:     {start_date}")
print(f"   • End date:       {end_date}")
print(f"   • Forecast days:  {output_len}")

# ===============================
# 🏃‍♂️‍➡️ Pipeline Run
# ===============================

if run_pipeline:
    pipeline = DataEngineeringPipeline(
        path_stock=path_stock,
        start_date=start_date,
        end_date=end_date,
        top_n_feature_important=10,
    )

    features, columns = pipeline.run()
    print("[INFO] Pipeline completed successfully.")

# worked with the reduced feratured engineering df

# Load datasets for each stock
# # Forecast horizon (e.g., predict 15 days ahead)

TSLA_df = pd.read_csv(
    f"{path_stock}/TSLA_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
NVDA_df = pd.read_csv(
    f"{path_stock}/NVDA_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
MSFT_df = pd.read_csv(
    f"{path_stock}/MSFT_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
GOOGL_df = pd.read_csv(
    f"{path_stock}/GOOGL_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
AMZN_df = pd.read_csv(
    f"{path_stock}/AMZN_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
AAPL_df = pd.read_csv(
    f"{path_stock}/AAPL_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
META_df = pd.read_csv(
    f"{path_stock}/META_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()

# Map stock tickers to their respective DataFrames
stock_dataframes = {
    "TSLA": TSLA_df,
    "NVDA": NVDA_df,
    "MSFT": MSFT_df,
    "GOOGL": GOOGL_df,
    "AMZN": AMZN_df,
    "AAPL": AAPL_df,
    "META": META_df,
}

# Select the correct DataFrame based on the target stock
model_df = stock_dataframes.get(target_stock)


# XXX: On Apple Silicon, Darts may require float32 instead of float64
# to avoid issues with PyTorch. This is a workaround for compatibility.
cols = model_df.select_dtypes(include=["float64"]).columns
model_df[cols] = model_df[cols].astype(np.float32)
# # Or alternatively, you can set the device to CPU

model_df_copy = model_df.copy(deep=True)


print("model_df_copy.columns :", model_df_copy.columns)
print("model_df_copy.index :", model_df_copy.index)
# Extract all feature column names, excluding the 'target' column
features_list_model = [col for col in model_df.columns if col != "target"]

# Print out the number of features and their names
print("\n[INFO] Number of features for model:", len(features_list_model))
print("\n[INFO] Feature list for model:", features_list_model)


# # Step 2-4:Reindexing to daily frequency


# ------------------------------------------------------------------------------
# 🧾 Step 2: Inspect & Standardize Time Index (Reindexing to Daily Frequency)
# ------------------------------------------------------------------------------
full_index = pd.date_range(
    start=model_df.index.min(), end=model_df.index.max(), freq="D"
)
if model_df.index.inferred_freq is None or not model_df.index.is_monotonic_increasing:
    model_df = model_df.reindex(full_index)
    model_df.interpolate(method="linear", limit_direction="both", inplace=True)
print("[INFO] Loaded model_dataset shaped: ", model_df.shape)
print("[INFO] Loaded model_dataset NaN: ", model_df.isnull().sum().sum())
print("[INFO] Loaded model_dataset type: ", type(model_df))
print(
    "[INFO] Loaded model_dataset index:",
    model_df.index.min(),
    "--- to ---",
    model_df.index.max(),
)


# # 🔁 Step 3: Convert to Darts TimeSeries format
#


# ---------------------------------------------
# 🎯 Step 3-1: Convert Target Column to Darts TimeSeries
# ---------------------------------------------
raw_target_df = model_df[["target"]]
raw_target = TimeSeries.from_dataframe(
    raw_target_df, value_cols="target", freq="D", fill_missing_dates=True
)
raw_covariates_df = model_df[features_list_model]
raw_covariates = TimeSeries.from_dataframe(
    raw_covariates_df, value_cols=features_list_model, freq="D", fill_missing_dates=True
)


# # Step 4: Time-based Split for Train/Val/Test


# -------------------------------------------------------------------------------------
# ✂️ Step 4: Time-based Split for Train/Val/Test
# -------------------------------------------------------------------------------------
from datetime import timedelta  # Used to offset time ranges for slicing covariates

output_len = output_len  # Number of days we want to predict into the future
input_len = (
    output_len * 5
)  # Number of past days (lookback window) to feed into the model

val_size = min(((input_len + output_len) + 5), int(0.1 * len(raw_target)))
train_target_raw = raw_target[: -val_size - output_len]
val_target_raw = raw_target[-val_size - output_len : -output_len]
test_target_raw = raw_target[-output_len:]

train_covariates_raw = raw_covariates.slice(
    train_target_raw.start_time(),  # Start of training period
    train_target_raw.end_time()
    + timedelta(days=output_len),  # Extend to include future forecast period
)

val_covariates_raw = raw_covariates.slice(
    val_target_raw.start_time(),  # Start of validation period
    val_target_raw.end_time()
    + timedelta(days=output_len),  # Extend to include forecast period
)

test_covariates_raw = raw_covariates.slice(
    test_target_raw.start_time()
    - timedelta(days=input_len),  # Start: back-projected from test start
    test_target_raw.end_time()
    + timedelta(days=output_len),  # End: includes forecast window
)


# # Step 5: Scale Target and Covariates


# --------------------------------------------------------------
# 🧮 Step 5: Normalize TimeSeries Data Using Darts Scaler
# --------------------------------------------------------------
t_scaler = Scaler()  # Target scaler
f_scaler = Scaler()  # Feature/covariates scaler

# Fit the target scaler on the training target only and transform all splits
train_target = t_scaler.fit_transform(
    train_target_raw
)  # Learn scaling on training target
val_target = t_scaler.transform(val_target_raw)  # Apply same scale to validation target
test_target = t_scaler.transform(test_target_raw)  # Apply same scale to test target
print("\n[INFO] Step 5🧾 target scaler Time Index Ranges after Scaling:")
print(
    f"[INFO] Train Target         : {train_target.time_index.min()} → {train_target.time_index.max()}"
)
print(
    f"[INFO] Validation Target    : {val_target.time_index.min()} → {val_target.time_index.max()}"
)
print(
    f"[INFO] Test Target          : {test_target.time_index.min()} → {test_target.time_index.max()}"
)

train_covariates = f_scaler.fit_transform(
    train_covariates_raw
)  # Learn scaling on training covariates
val_covariates = f_scaler.transform(
    val_covariates_raw
)  # Apply same scale to validation covariates
test_covariates = f_scaler.transform(
    test_covariates_raw
)  # Apply same scale to test covariates

print("\n[INFO] Step 5🧾 covariate scaler Time Index Ranges after Scaling:")
print(
    f"[INFO] Train Covariates    : {train_covariates.time_index.min()} → {train_covariates.time_index.max()}"
)
print(
    f"[INFO] Validation Covariates: {val_covariates.time_index.min()} → {val_covariates.time_index.max()}"
)
print(
    f"[INFO] Test Covariates      : {test_covariates.time_index.min()} → {test_covariates.time_index.max()}"
)


# # Step 6: Initialize and Train TFT Model
#


# -------------------------------------------------------------------------------------
# 🧠 Step 6: Initialize and Train TFT Model
# -------------------------------------------------------------------------------------

# Initialize the Temporal Fusion Transformer (TFT) model from Darts
from signal_sigma.core.loss_history import LossHistory

loss_logger = LossHistory()  # Instantiate the callback
model = TFTModel(
    input_chunk_length=input_len,  # 🔁 Number of historical time steps used as input
    output_chunk_length=output_len,  # 🔮 Number of future time steps to predict
    hidden_size=64,  # 💡 Number of hidden units in LSTM layers (feature extractor size)
    lstm_layers=1,  # 🔄 Number of LSTM layers used in the encoder-decoder architecture
    dropout=0.1,  # 🕳 Dropout rate for regularization (to prevent overfitting)
    batch_size=64,  # 📦 Number of samples per training batch
    n_epochs=2,  # 🔁 Number of training epochs
    num_attention_heads=1,  # 🎯 Heads in multi-head attention layer for learning temporal patterns
    force_reset=True,  # 🧽 Force fresh training, resetting previous weights and checkpoints
    save_checkpoints=True,  # 💾 Save intermediate models automatically during training
    # 🎯 Quantile regression for probabilistic forecasting (predicts intervals)
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    # 🗓️ Add time-based encoders to give model temporal awareness
    add_encoders={
        "cyclic": {
            "past": [
                "month",
                "day",
                "weekday",
            ]  # Cyclical encodings (e.g., sin/cos for months, days)
        },
        "datetime_attribute": {
            "past": ["year", "month", "weekday"]  # Raw datetime attributes
        },
        "position": {
            "past": ["relative"]  # Relative time encoding (position in sequence)
        },
    },
    # ⚙️ Pass advanced settings to PyTorch Lightning trainer
    # ⚙️ Pass advanced settings to PyTorch Lightning trainer
    pl_trainer_kwargs={
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5),  # ⏸ Early stopping callback
            loss_logger,  # 📊 Our custom loss tracking callback
        ],
        "log_every_n_steps": 10,  # 📝 How often to log training steps
        "enable_model_summary": True,  # 📋 Show model layer summary
        "enable_progress_bar": True,  # 📊 Show progress bar during training
        "logger": True,  # 🧾 Enable default logger (e.g., TensorBoard)
    },
)


# ---------------------------------------------------------------
# 🚀 Fit (Train) the Model
# ---------------------------------------------------------------

model.fit(
    series=train_target,  # 🎯 Target series for training
    future_covariates=train_covariates,  # 🔮 Associated features for training (aligned with future steps)
    val_series=val_target,  # 🎯 Validation target series (to monitor overfitting)
    val_future_covariates=val_covariates,  # 🔮 Validation covariates aligned with val_target
    verbose=True,  # 📣 Print training progress to console
)


# # Step 7: Plot Train vs Validation Loss Over Epochs


# -------------------------------------------------------------------------------------
# Step 7: Plot Train vs Validation Loss Over Epochs
# -------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(loss_logger.epochs, loss_logger.train_losses, label="Train Loss", marker="o")
plt.plot(
    loss_logger.epochs, loss_logger.val_losses, label="Validation Loss", marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title(F"Train vs Validation Loss Over Epochs -\n Feature:{features_list_model}")
plt.title("Train vs Validation Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# #  Single-Shot Forecast for Singel Point and Quantiles


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# === Step 1: Predict quantile forecasts (probabilistic output with uncertainty) ===
quantil_forecast = model.predict(
    n=output_len,  # Number of time steps to forecast
    series=val_target,  # Past target values to condition the model
    future_covariates=test_covariates,  # Known future inputs (calendar features, macro, etc.)
    num_samples=100,  # Enables Monte Carlo sampling for quantiles
)

# === Step 2: Predict deterministic (point) forecast separately ===
point_forecast = model.predict(
    n=output_len, series=val_target, future_covariates=test_covariates
)

# === Step 3: Inverse transform all forecast outputs back to original scale ===
quantil_forecast_p10 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.1)
)
quantil_forecast_p50 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.5)
)
quantil_forecast_p90 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.9)
)
point_forecast_inv = t_scaler.inverse_transform(point_forecast)

# === Step 4: Align forecasts and ground truth by shared date index ===
common_index = quantil_forecast_p50.time_index.intersection(test_target_raw.time_index)

# Extract aligned values as 1D arrays
true_vals = test_target_raw[common_index].values().squeeze()
p10_vals = quantil_forecast_p10[common_index].values().squeeze()
p50_vals = quantil_forecast_p50[common_index].values().squeeze()
p90_vals = quantil_forecast_p90[common_index].values().squeeze()
point_vals = point_forecast_inv[common_index].values().squeeze()
time_index = common_index


# === Step 5: Define evaluation metrics as helper functions ===
def smape(y_true, y_pred):
    return 200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)


def mape(y_true, y_pred):
    return 100.0 * np.abs((y_true - y_pred) / (y_true + 1e-9))


def pinball_loss(y_true, y_pred, q):
    delta = y_true - y_pred
    return np.maximum(q * delta, (q - 1) * delta)


# === Step 6: Initialize evaluation DataFrame ===
forecast_df = pd.DataFrame(
    {
        "date": time_index,
        "true": true_vals,  # Ground truth
        "p10": p10_vals,  # 10% lower quantile
        "p50": p50_vals,  # Median forecast
        "p90": p90_vals,  # 90% upper quantile
        "point_forecast": point_vals,  # Point (mean or deterministic) prediction
    }
)

# === Step 7: Compute residuals and interval width ===
forecast_df["residual_point"] = forecast_df["true"] - forecast_df["point_forecast"]
forecast_df["residual_p50"] = forecast_df["true"] - forecast_df["p50"]
forecast_df["quantile_width_80"] = forecast_df["p90"] - forecast_df["p10"]
forecast_df["z_residual_point"] = forecast_df["residual_point"] / (
    forecast_df["quantile_width_80"] + 1e-9
)

# === Step 8A: Point forecast metrics (per day) ===
forecast_df["mae_point"] = np.abs(forecast_df["residual_point"])
forecast_df["mape_point"] = mape(forecast_df["true"], forecast_df["point_forecast"])
forecast_df["smape_point"] = smape(forecast_df["true"], forecast_df["point_forecast"])
forecast_df["rmse_point"] = np.sqrt(forecast_df["residual_point"] ** 2)
forecast_df["r2_point"] = r2_score(forecast_df["true"], forecast_df["point_forecast"])

# === Step 8B: Quantile p50 forecast metrics (for comparison) ===
forecast_df["mae_p50"] = np.abs(forecast_df["residual_p50"])
forecast_df["mape_p50"] = mape(forecast_df["true"], forecast_df["p50"])
forecast_df["smape_p50"] = smape(forecast_df["true"], forecast_df["p50"])
forecast_df["rmse_p50"] = np.sqrt(forecast_df["residual_p50"] ** 2)
forecast_df["r2_p50"] = r2_score(forecast_df["true"], forecast_df["p50"])

# === Step 9: Pinball loss per quantile (measures sharpness + calibration) ===
forecast_df["pinball_p10"] = pinball_loss(forecast_df["true"], forecast_df["p10"], 0.1)
forecast_df["pinball_p50"] = pinball_loss(forecast_df["true"], forecast_df["p50"], 0.5)
forecast_df["pinball_p90"] = pinball_loss(forecast_df["true"], forecast_df["p90"], 0.9)

# === Step 10: Interval coverage (binary flag per day) ===
forecast_df["interval_covered_80"] = (
    (forecast_df["true"] >= forecast_df["p10"])
    & (forecast_df["true"] <= forecast_df["p90"])
).astype(int)

# === Step 11: Quantile calibration columns — 1 if true <= quantile prediction ===
quantiles = np.linspace(0.05, 0.95, 19)
empirical_coverage_dict = {
    f"coverage_q{int(q * 100)}": (
        forecast_df["true"]
        <= quantil_forecast.quantile_timeseries(q).values().squeeze()
    ).astype(int)
    for q in quantiles
}
# Add each column to the main DataFrame
for col, values in empirical_coverage_dict.items():
    forecast_df[col] = values

# === Step 12: Final formatting and export ===
# Step 1: Make sure forecast_df index is set to date
forecast_df.set_index("date", inplace=True)
forecast_df.to_csv(
    f"{path_stock}/{target_stock}_forecast_eval_metrics_{start_date}_{end_date}.csv"
)

# Step 2: Perform a left join on time index
model_result_df = model_df_copy.join(forecast_df, how="left")

# Step 3: Fill missing predictions (non-forecasted rows) with 0
model_result_df.fillna(0, inplace=True)
# Step 4: Save to CSV
model_result_df.to_csv(
    f"{path_stock}/{target_stock}_Model_Initial_Result_{start_date}_{end_date}.csv"
)

print("forecast_df.columns :", forecast_df.columns)
print("model_result_df.columns :", model_result_df.columns)

# Optionally inspect
model_result_df.tail(5)


# # Step 11: Plot Forecast vs Actual (Quantiles + True)


# === Step 1: Set start of plot range for history context ===
plot_start = Timestamp("2024-10-01")
raw_target_slice = raw_target.slice(plot_start, raw_target.end_time())
full_target_index = raw_target_slice.time_index
full_target_vals = raw_target_slice.values().squeeze()

# === Step 2: Extract forecast and truth values from forecast_df ===
forecast_index = forecast_df.index
y_true = forecast_df["true"].astype(float).values
y_p10 = forecast_df["p10"].astype(float).values
y_p50 = forecast_df["p50"].astype(float).values
y_p90 = forecast_df["p90"].astype(float).values
y_point = forecast_df["point_forecast"].astype(float).values

# === Step 3: Define split markers ===
# Assuming you have already split your raw_target like this:

train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = test_target_raw.start_time()


train_target_raw = raw_target[:train_end]
val_target_raw = raw_target[train_end:val_end]
test_target_raw = raw_target[val_end:]

train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = forecast_index[0]

# === Step 4: Begin plot ===
plt.figure(figsize=(14, 6))

# Plot full historical target
plt.plot(
    full_target_index,
    full_target_vals,
    label="Historical Target",
    color="black",
    linewidth=2,
)

# Plot true values in test set
plt.plot(forecast_index, y_true, label="True (Test)", color="blue", linewidth=2)

# Plot forecasts
plt.plot(
    forecast_index, y_p50, label="Forecast Median (p50)", color="magenta", linewidth=2
)
plt.plot(
    forecast_index, y_p10, label="Forecast Lower (p10)", linestyle="--", color="skyblue"
)
plt.plot(
    forecast_index, y_p90, label="Forecast Upper (p90)", linestyle="--", color="green"
)
plt.plot(forecast_index, y_point, label="Forecast Point", linestyle=":", color="orange")

# Shaded p10–p90 interval
plt.fill_between(
    forecast_index,
    y_p10,
    y_p90,
    color="lightgray",
    alpha=0.4,
    label="Confidence Interval (p10–p90)",
)

# === Step 5: Add vertical split markers ===
plt.axvline(train_end, color="gray", linestyle="--", linewidth=1.5, label="Train End")
plt.axvline(
    val_end, color="purple", linestyle="--", linewidth=1.5, label="Validation End"
)
plt.axvline(
    test_start, color="red", linestyle="--", linewidth=2, label="Forecast Start"
)

# === Style ===
plt.title(f"TFT Forecast vs. True - Full Signal View ({target_stock})")
plt.xlabel("Date")
plt.ylabel("Target Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from pandas import Timestamp

# === Step 1: Set start of plot range for history context ===
plot_start = Timestamp("2025-01-01")
raw_target_slice = raw_target.slice(plot_start, raw_target.end_time())
full_target_index = raw_target_slice.time_index
full_target_vals = raw_target_slice.values().squeeze()

# === Step 2: Extract forecast and truth values from forecast_df ===
forecast_index = forecast_df.index
y_true = forecast_df["true"].astype(float).values
y_p10 = forecast_df["p10"].astype(float).values
y_p50 = forecast_df["p50"].astype(float).values
y_p90 = forecast_df["p90"].astype(float).values
y_point = forecast_df["point_forecast"].astype(float).values

# === Step 3: Define split markers ===
train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = forecast_index[0]

# === Step 4: Begin plot ===
fig, ax = plt.subplots(figsize=(14, 7))

# Plot full historical target
ax.plot(
    full_target_index,
    full_target_vals,
    label="Historical Target",
    color="black",
    linewidth=2,
)

# Plot true values in test set
ax.plot(forecast_index, y_true, label="True (Test)", color="blue", linewidth=2)

# Plot forecasts
ax.plot(
    forecast_index, y_p50, label="Forecast Median (p50)", color="magenta", linewidth=2
)
ax.plot(
    forecast_index, y_p10, label="Forecast Lower (p10)", linestyle="--", color="skyblue"
)
ax.plot(
    forecast_index, y_p90, label="Forecast Upper (p90)", linestyle="--", color="green"
)
ax.plot(forecast_index, y_point, label="Forecast Point", linestyle=":", color="orange")

# Confidence interval shading
ax.fill_between(
    forecast_index,
    y_p10,
    y_p90,
    color="lightgray",
    alpha=0.4,
    label="Confidence Interval (p10–p90)",
)

# Add vertical markers
ax.axvline(train_end, color="gray", linestyle="--", linewidth=1.5, label="Train End")
ax.axvline(
    val_end, color="purple", linestyle="--", linewidth=1.5, label="Validation End"
)
ax.axvline(test_start, color="red", linestyle="--", linewidth=2, label="Forecast Start")

# === Annotations for Min/Max in Test Predictions ===
min_val = y_true.min()
min_idx = forecast_index[y_true.argmin()]
ax.annotate(
    f"Min: {min_val:.2f}",
    xy=(min_idx, min_val),
    xytext=(min_idx, min_val - 10),
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=10,
    color="red",
)

max_val = y_true.max()
max_idx = forecast_index[y_true.argmax()]
ax.annotate(
    f"Max: {max_val:.2f}",
    xy=(max_idx, max_val),
    xytext=(max_idx, max_val + 10),
    arrowprops=dict(facecolor="green", shrink=0.05),
    fontsize=10,
    color="green",
)

# === Titles and Labels ===
ax.set_title(f"TFT Forecast vs. True - Full Signal View ({target_stock})", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Target Value")
ax.grid(True)

# === Horizontal Legend at Bottom ===
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from pandas import Timestamp

# === Step 1: Set start of plot range for history context ===
plot_start = Timestamp("2018-01-01")
raw_target_slice = raw_target.slice(plot_start, raw_target.end_time())
full_target_index = raw_target_slice.time_index
full_target_vals = raw_target_slice.values().squeeze()

# === Step 2: Extract forecast and truth values from forecast_df ===
forecast_index = forecast_df.index
y_true = forecast_df["true"].astype(float).values
y_p10 = forecast_df["p10"].astype(float).values
y_p50 = forecast_df["p50"].astype(float).values
y_p90 = forecast_df["p90"].astype(float).values
y_point = forecast_df["point_forecast"].astype(float).values

# === Step 3: Define split markers ===
train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = forecast_index[0]

# === Step 4: Begin plot ===
fig, ax = plt.subplots(figsize=(14, 7))

# Plot full historical target
ax.plot(
    full_target_index,
    full_target_vals,
    label="Historical Target",
    color="black",
    linewidth=2,
)

# Plot true values in test set
ax.plot(forecast_index, y_true, label="True (Test)", color="blue", linewidth=2)

# Plot forecasts
ax.plot(
    forecast_index, y_p50, label="Forecast Median (p50)", color="magenta", linewidth=2
)
ax.plot(
    forecast_index, y_p10, label="Forecast Lower (p10)", linestyle="--", color="skyblue"
)
ax.plot(
    forecast_index, y_p90, label="Forecast Upper (p90)", linestyle="--", color="green"
)
ax.plot(forecast_index, y_point, label="Forecast Point", linestyle=":", color="orange")

# Confidence interval shading
ax.fill_between(
    forecast_index,
    y_p10,
    y_p90,
    color="lightgray",
    alpha=0.4,
    label="Confidence Interval (p10–p90)",
)

# Add vertical markers
ax.axvline(train_end, color="gray", linestyle="--", linewidth=1.5, label="Train End")
ax.axvline(
    val_end, color="purple", linestyle="--", linewidth=1.5, label="Validation End"
)
ax.axvline(test_start, color="red", linestyle="--", linewidth=2, label="Forecast Start")

# === Annotations for Min/Max in Test Predictions ===
min_val = y_true.min()
min_idx = forecast_index[y_true.argmin()]
ax.annotate(
    f"Min: {min_val:.2f}",
    xy=(min_idx, min_val),
    xytext=(min_idx, min_val - 10),
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=10,
    color="red",
)

max_val = y_true.max()
max_idx = forecast_index[y_true.argmax()]
ax.annotate(
    f"Max: {max_val:.2f}",
    xy=(max_idx, max_val),
    xytext=(max_idx, max_val + 10),
    arrowprops=dict(facecolor="green", shrink=0.05),
    fontsize=10,
    color="green",
)

# === Titles and Labels ===
ax.set_title(f"TFT Forecast vs. True - Full Signal View ({target_stock})", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Target Value")
ax.grid(True)

# === Horizontal Legend at Bottom ===
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Aggregate Metrics ===
avg_metrics = {
    "R² (Point)": forecast_df["r2_point"].mean(),
    "R² (p50)": forecast_df["r2_p50"].mean(),
    "MAE (Point)": forecast_df["mae_point"].mean(),
    "MAE (p50)": forecast_df["mae_p50"].mean(),
    "Pinball (p10)": forecast_df["pinball_p10"].mean(),
    "Pinball (p50)": forecast_df["pinball_p50"].mean(),
    "Pinball (p90)": forecast_df["pinball_p90"].mean(),
    "Interval Covered (80%)": forecast_df["interval_covered_80"].mean(),
}

# === Step 2: Plot ===
plt.figure(figsize=(12, 6))
colors = ["orange", "orange", "blue", "blue", "green", "green", "green", "purple"]
bars = plt.bar(avg_metrics.keys(), avg_metrics.values(), color=colors)

# Annotate values on bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

# Style
plt.title(
    f"Average Forecast Metrics (Point, Quantiles, Interval) - ({target_stock})",
    fontsize=14,
)
plt.ylabel("Metric Value")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# === Step 1: Extract relevant series ===
dates = forecast_df.index
mae_p50 = forecast_df["mae_p50"]
pinball_p10 = forecast_df["pinball_p10"]
pinball_p50 = forecast_df["pinball_p50"]
pinball_p90 = forecast_df["pinball_p90"]
interval_covered_80 = forecast_df["interval_covered_80"]

# === Step 2: Subplots layout ===
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

# --- Plot 1: MAE (p50) ---
axs[0].plot(dates, mae_p50, label="MAE (p50)", color="green", linewidth=2)
axs[0].set_ylabel("MAE")
axs[0].set_title(f"Mean Absolute Error for Median Forecast (p50)  --  ({target_stock})")
axs[0].grid(True)
axs[0].legend(loc="upper right")

# --- Plot 2: Pinball Losses ---
axs[1].plot(
    dates, pinball_p10, label="Pinball Loss (p10)", color="skyblue", linewidth=2
)
axs[1].plot(dates, pinball_p50, label="Pinball Loss (p50)", color="purple", linewidth=2)
axs[1].plot(dates, pinball_p90, label="Pinball Loss (p90)", color="brown", linewidth=2)
axs[1].set_ylabel("Pinball Loss")
axs[1].set_title(f"Pinball Loss for Quantile Forecasts -- ({target_stock})")
axs[1].grid(True)
axs[1].legend(loc="upper right")

# --- Plot 3: Interval Coverage ---
axs[2].plot(
    dates,
    interval_covered_80,
    label="Interval Covered (80%)",
    color="black",
    linestyle="--",
    linewidth=2,
)
axs[2].set_ylabel("Coverage")
axs[2].set_title(f"80% Prediction Interval Coverage -- ({target_stock})")
axs[2].set_xlabel("Date")
axs[2].grid(True)
axs[2].legend(loc="upper right")

# === Final layout ===
plt.tight_layout()
plt.show()
