# ---
# description: Processes data from FRED.
# ---

# ---

import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


class FredMacroProcessor:
    """
    📊 FredMacroProcessor automates the retrieval, scaling, and synthesis of key macroeconomic indicators
    from the FRED (Federal Reserve Economic Data) API.

    ➤ Final output includes 3 interpretable, economically grounded composite features:
        1. inflation_monetary_pressure
        2. labor_econ_activity
        3. consumer_spending_sentiment
    """

    def __init__(self, start_date="2000-01-01", save_path="../data/fed"):
        # Load API key from .env file
        load_dotenv()
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Please set the FRED_API_KEY environment variable.")

        self.start_date = pd.to_datetime(start_date)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Define FRED series to use
        self.indicators = {
            "cpi": "CPIAUCSL",
            "core_cpi": "CPILFESL",
            "pce": "PCEPI",
            "core_pce": "PCEPILFE",
            "fed_funds_rate": "FEDFUNDS",
            "unemployment_rate": "UNRATE",
            "nonfarm_payrolls": "PAYEMS",
            "retail_sales": "RSXFS",
            "consumer_sentiment": "UMCSENT",
            "housing_starts": "HOUST",
            "10y_treasury_yield": "DGS10",
            "industrial_production": "INDPRO",
            "real_personal_income": "W875RX1",
            "initial_jobless_claims": "ICSA",
        }

    def fetch_data(self) -> pd.DataFrame:
        """
        📡 Fetch macroeconomic indicators from FRED, resample to daily frequency,
        and apply forward/backward fill for missing values.
        """
        print("📡 Fetching macroeconomic indicators from FRED...")
        fred = Fred(api_key=self.api_key)
        df = pd.DataFrame()

        for name, series_id in self.indicators.items():
            print(f"→ Fetching: {name} ({series_id})")
            try:
                series = fred.get_series(series_id)
                series = series[series.index >= self.start_date]
                series = series.resample("D").ffill()  # Daily resolution
                df[name] = series
            except Exception as e:
                print(f"⚠️ Error fetching {name}: {e}")

        # Fill missing values
        split_date = pd.to_datetime("2016-01-01")
        df.loc[df.index >= split_date] = df.loc[df.index >= split_date].ffill()
        df.loc[df.index < split_date] = df.loc[df.index < split_date].bfill()

        return df

    def preprocess_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🧼 Normalize macroeconomic features with Z-score scaling before combining.
        """
        print("🔄 Scaling raw macro features with StandardScaler...")
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
        return scaled_df

    def create_composites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🧮 Create composite macro features from scaled input.
        """
        # Normalize inverse-based indicators to align directionality
        df["inv_unemployment"] = 1 / (df["unemployment_rate"] + 1e-3)
        df["inv_claims"] = 1 / (df["initial_jobless_claims"] + 1e-3)

        # Re-scale the added inverse columns
        scaler = StandardScaler()
        df[["inv_unemployment", "inv_claims"]] = scaler.fit_transform(
            df[["inv_unemployment", "inv_claims"]]
        )

        # 🔵 1. Inflation & Monetary Pressure
        df["FRED_inflation_monetary_pressure"] = (
            0.25 * df["cpi"]
            + 0.20 * df["core_cpi"]
            + 0.15 * df["pce"]
            + 0.15 * df["core_pce"]
            + 0.15 * df["fed_funds_rate"]
            + 0.10 * df["10y_treasury_yield"]
        )

        # 🟢 2. Labor & Economic Activity
        df["FRED_labor_econ_activity"] = (
            0.35 * df["nonfarm_payrolls"]
            + 0.25 * df["inv_unemployment"]
            + 0.20 * df["inv_claims"]
            + 0.20 * df["industrial_production"]
        )

        # 🟡 3. Consumer Sentiment & Spending
        df["FRED_consumer_spending_sentiment"] = (
            0.35 * df["retail_sales"]
            + 0.25 * df["real_personal_income"]
            + 0.25 * df["consumer_sentiment"]
            + 0.15 * df["housing_starts"]
        )

        return df[
            [
                "FRED_inflation_monetary_pressure",
                "FRED_labor_econ_activity",
                "FRED_consumer_spending_sentiment",
            ]
        ]

    def run_pipeline(self) -> pd.DataFrame:
        """
        🚀 Execute full FRED macro feature pipeline:
        - Fetch → Scale → Create Composites → Save → Return
        """
        raw_df = self.fetch_data()
        scaled_input = self.preprocess_and_scale(raw_df)
        composite_df = self.create_composites(scaled_input)

        composite_df.to_csv(f"{self.save_path}/fred_macro_composites.csv")
        print(
            f"\n✅ Saved scaled macro composite features to {self.save_path}/fred_macro_composites.csv"
        )

        return composite_df
