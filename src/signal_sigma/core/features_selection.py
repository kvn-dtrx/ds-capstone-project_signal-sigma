# ---
# description: Selects features from data automatically.
# ---

# ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ReducedFeatureSelector:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        top_n: int = 15,
        stock_name: str = "UNKNOWN",
    ):
        self.df = data.dropna().copy()
        self.target_col = target_col
        self.top_n = top_n
        self.stock_name = stock_name
        self.X_raw = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        self.feature_weights = {}
        self.model_weights = {
            "Pearson": 0.15,
            "MutualInfo": 0.15,
            "XGBoost": 0.40,
            "RandomForest": 0.30,
        }
        self.scaler_full = StandardScaler()
        self.X_scaled_full = pd.DataFrame(
            self.scaler_full.fit_transform(self.X_raw),
            columns=self.X_raw.columns,
            index=self.X_raw.index,
        )

    def calculate_metrics(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        }

    def plot_importances(self, importances: pd.Series, title: str):
        top_features = importances.sort_values(ascending=False).head(self.top_n)
        plt.figure(figsize=(6, 3))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f"{self.stock_name} - {title} (Top {self.top_n})")
        plt.tight_layout()
        plt.show()

    def add_weighted_scores(self, importances: pd.Series, model_name: str):
        norm = (importances - importances.min()) / (
            importances.max() - importances.min() + 1e-9
        )
        weighted_scores = norm * self.model_weights[model_name]
        for feature, score in weighted_scores.items():
            self.feature_weights[feature] = self.feature_weights.get(feature, 0) + score
        return importances

    def select_features(self):
        metrics_dict = {}
        # --- Pearson Correlation ---
        pearson = (
            self.X_scaled_full.corrwith(self.y)
            .abs()
            .sort_values(ascending=False)
            .head(self.top_n)
        )
        self.plot_importances(pearson, "Pearson Correlation")
        self.add_weighted_scores(pearson, "Pearson")
        # --- Mutual Information ---
        mi = mutual_info_regression(self.X_scaled_full, self.y, random_state=42)
        mi_series = (
            pd.Series(mi, index=self.X_scaled_full.columns)
            .sort_values(ascending=False)
            .head(self.top_n)
        )
        self.plot_importances(mi_series, "Mutual Information")
        self.add_weighted_scores(mi_series, "MutualInfo")
        # --- Train/Test Split ---
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            self.X_raw, self.y, test_size=0.1, random_state=42
        )
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=X_train_raw.columns,
            index=X_train_raw.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=X_test_raw.columns,
            index=X_test_raw.index,
        )
        # --- XGBoost ---
        xgb = XGBRegressor(n_estimators=300, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_importance = (
            pd.Series(xgb.feature_importances_, index=X_train.columns)
            .sort_values(ascending=False)
            .head(self.top_n)
        )
        self.plot_importances(xgb_importance, "XGBoost Importance")
        self.add_weighted_scores(xgb_importance, "XGBoost")
        metrics_dict["XGBoost"] = self.calculate_metrics(y_test, xgb.predict(X_test))
        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        rf_importance = (
            pd.Series(rf.feature_importances_, index=X_train.columns)
            .sort_values(ascending=False)
            .head(self.top_n)
        )
        self.plot_importances(rf_importance, "Random Forest Importance")
        self.add_weighted_scores(rf_importance, "RandomForest")
        metrics_dict["RandomForest"] = self.calculate_metrics(
            y_test, rf.predict(X_test)
        )
        # --- Final Weighted Voting Result ---
        weighted_feature_df = pd.DataFrame.from_dict(
            self.feature_weights, orient="index", columns=["WeightedScore"]
        )
        weighted_feature_df = weighted_feature_df.sort_values(
            "WeightedScore", ascending=False
        )
        print(
            "\n:white_tick: Final Weighted Feature Importance (Merged from All Methods):"
        )
        print(weighted_feature_df.head(self.top_n))
        print("\n:bar_chart: Model Performance on Test Set:")
        for name, metrics in metrics_dict.items():
            print(f"{name} Metrics: {metrics}")
        return weighted_feature_df, metrics_dict
