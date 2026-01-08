# ---
# title: TemporalFeatureCombiner Class
# ---

# ---

import pandas as pd
from sklearn.preprocessing import StandardScaler


class TemporalFeatureCombiner:
    """
    Combine and compress time series features into short-, medium-, and long-term signals.
    All input features are scaled before combination for statistical consistency.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def combine(self):
        df = self.df.copy()

        # ⚖️ Apply scaling to all raw inputs before combining
        raw_feature_cols = [
            "trend_macd",
            "trend_ema_50",
            "trend_ema_100",
            "trend_macd_signal",
            "volume_obv",
            "volume_cmf",
            "volume_vpt",
            "lag_14",
            "diff_14",
            "trend_sma_100",
            "trend_ema_250",
            "volume_ad",
            "lag_35",
            "diff_35",
            "trend_sma_300",
            "trend_ema_1000",
            "trend_sma_650",
            "volatility_donchian_high_45",
            "volatility_donchian_low_45",
            "lag_100",
            "diff_100",
        ]

        # Drop missing raw features safely and scale
        available_cols = [col for col in raw_feature_cols if col in df.columns]
        scaler = StandardScaler()
        df[available_cols] = scaler.fit_transform(df[available_cols])

        # --- Short-Term Signals (3–25 days) ---
        df["TI_short_trend_signal"] = (
            0.4 * df.get("trend_macd", 0)
            + 0.3 * (df.get("trend_ema_50", 0) - df.get("trend_ema_100", 0))
            + 0.3 * df.get("trend_macd_signal", 0)
        )

        df["TI_short_volume_pressure"] = (
            0.4 * df.get("volume_obv", 0)
            + 0.4 * df.get("volume_cmf", 0)
            + 0.2 * df.get("volume_vpt", 0)
        )

        df["TI_short_lag_movement"] = 0.5 * df.get("lag_14", 0) + 0.5 * df.get(
            "diff_14", 0
        )

        # --- Medium-Term Signals (25–100 days) ---
        df["TI_medium_trend_consistency"] = (
            0.3 * df.get("trend_ema_100", 0)
            + 0.3 * df.get("trend_sma_100", 0)
            + 0.4 * df.get("trend_ema_250", 0)
        )

        df["TI_medium_volume_pressure"] = (
            0.4 * df.get("volume_cmf", 0)
            + 0.3 * df.get("volume_ad", 0)
            + 0.3 * df.get("volume_vpt", 0)
        )

        df["TI_medium_lag_movement"] = 0.5 * df.get("lag_35", 0) + 0.5 * df.get(
            "diff_35", 0
        )

        # --- Long-Term Signals (100+ days) ---
        df["TI_long_term_trend_strength"] = (
            0.4 * df.get("trend_sma_300", 0)
            + 0.3 * df.get("trend_ema_1000", 0)
            + 0.3 * df.get("trend_sma_650", 0)
        )

        df["TI_long_donchian_channel_width"] = df.get(
            "volatility_donchian_high_45", 0
        ) - df.get("volatility_donchian_low_45", 0)

        df["TI_long_lag_signal"] = 0.5 * df.get("lag_100", 0) + 0.5 * df.get(
            "diff_100", 0
        )

        self.df = df
        return df
