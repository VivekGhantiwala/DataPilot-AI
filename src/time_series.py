"""
Time Series Analysis Module
Provides comprehensive time series analysis, forecasting, and anomaly detection.
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try importing statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Try importing prophet
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class TimeSeriesAnalyzer:
    """
    Comprehensive Time Series Analysis and Forecasting.

    Features:
    - Trend and seasonality decomposition
    - Stationarity testing (ADF, KPSS)
    - Autocorrelation analysis
    - Multiple forecasting methods (ARIMA, ETS, Prophet)
    - Anomaly detection
    - Performance metrics

    Example:
        >>> ts = TimeSeriesAnalyzer(data, date_column='date', value_column='sales')
        >>> ts.analyze()
        >>> forecast = ts.forecast(periods=30)
        >>> ts.plot_forecast()
    """

    def __init__(self, data: pd.DataFrame, date_column: str, value_column: str, freq: Optional[str] = None):
        """
        Initialize the TimeSeriesAnalyzer.

        Args:
            data: DataFrame with time series data
            date_column: Name of the date/datetime column
            value_column: Name of the value column
            freq: Frequency string (e.g., 'D', 'W', 'M', 'H')
        """
        self.data = data.copy()
        self.date_column = date_column
        self.value_column = value_column

        # Prepare time series
        self._prepare_data()

        # Infer frequency if not provided
        self.freq = freq or self._infer_frequency()

        # Analysis results
        self.analysis_results = {}
        self.decomposition = None
        self.forecast_results = {}
        self.best_model = None
        self.best_model_name = None

        # Check dependencies
        if not STATSMODELS_AVAILABLE:
            print("⚠️ statsmodels not installed. Install with: pip install statsmodels")

    def _prepare_data(self):
        """Prepare data for time series analysis."""
        # Convert to datetime
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        # Sort by date
        self.data = self.data.sort_values(self.date_column).reset_index(drop=True)

        # Create time series
        self.ts = self.data.set_index(self.date_column)[self.value_column]

        # Handle missing values
        if self.ts.isnull().any():
            self.ts = self.ts.interpolate(method="linear")

        print(f"✅ Time series prepared: {len(self.ts)} observations")
        print(f"   Date range: {self.ts.index.min()} to {self.ts.index.max()}")

    def _infer_frequency(self) -> str:
        """Infer the frequency of the time series."""
        if len(self.ts) < 2:
            return "D"

        # Calculate median time difference
        diffs = pd.Series(self.ts.index).diff().dropna()
        median_diff = diffs.median()

        if median_diff <= timedelta(hours=1):
            return "H"
        elif median_diff <= timedelta(days=1):
            return "D"
        elif median_diff <= timedelta(days=7):
            return "W"
        elif median_diff <= timedelta(days=31):
            return "M"
        else:
            return "Y"

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics of the time series.

        Returns:
            Dictionary with basic statistics
        """
        stats = {
            "count": len(self.ts),
            "mean": float(self.ts.mean()),
            "std": float(self.ts.std()),
            "min": float(self.ts.min()),
            "max": float(self.ts.max()),
            "median": float(self.ts.median()),
            "range": float(self.ts.max() - self.ts.min()),
            "cv": float(self.ts.std() / self.ts.mean()) if self.ts.mean() != 0 else 0,
            "start_date": str(self.ts.index.min()),
            "end_date": str(self.ts.index.max()),
            "frequency": self.freq,
        }
        return stats

    def test_stationarity(self) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests.

        Returns:
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for stationarity tests.")

        results = {}

        # ADF Test
        adf_result = adfuller(self.ts.dropna())
        results["adf"] = {
            "test_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "is_stationary": adf_result[1] < 0.05,
        }

        # KPSS Test
        try:
            kpss_result = kpss(self.ts.dropna(), regression="c")
            results["kpss"] = {
                "test_statistic": float(kpss_result[0]),
                "p_value": float(kpss_result[1]),
                "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
                "is_stationary": kpss_result[1] > 0.05,
            }
        except Exception as e:
            results["kpss"] = {"error": str(e)}

        # Overall assessment
        if results["adf"]["is_stationary"] and results.get("kpss", {}).get("is_stationary", True):
            results["conclusion"] = "stationary"
        elif not results["adf"]["is_stationary"] and not results.get("kpss", {}).get("is_stationary", True):
            results["conclusion"] = "non-stationary"
        else:
            results["conclusion"] = "inconclusive"

        self.analysis_results["stationarity"] = results
        return results

    def decompose(self, model: str = "additive", period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)

        Returns:
            Dictionary with decomposition components
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for decomposition.")

        # Auto-detect period
        if period is None:
            if self.freq == "H":
                period = 24
            elif self.freq == "D":
                period = 7
            elif self.freq == "W":
                period = 52
            elif self.freq == "M":
                period = 12
            else:
                period = 4

        # Ensure we have enough data
        if len(self.ts) < 2 * period:
            period = len(self.ts) // 2

        self.decomposition = seasonal_decompose(self.ts.dropna(), model=model, period=period)

        components = {
            "trend": self.decomposition.trend,
            "seasonal": self.decomposition.seasonal,
            "residual": self.decomposition.resid,
            "observed": self.decomposition.observed,
        }

        self.analysis_results["decomposition"] = {
            "model": model,
            "period": period,
            "trend_strength": float(
                1 - (components["residual"].var() / (components["trend"] + components["residual"]).var())
            ),
            "seasonal_strength": float(
                1 - (components["residual"].var() / (components["seasonal"] + components["residual"]).var())
            ),
        }

        return components

    def analyze_autocorrelation(self, nlags: int = 40) -> Dict[str, Any]:
        """
        Analyze autocorrelation and partial autocorrelation.

        Args:
            nlags: Number of lags to analyze

        Returns:
            Dictionary with ACF and PACF values
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for autocorrelation analysis.")

        ts_clean = self.ts.dropna()
        nlags = min(nlags, len(ts_clean) // 2 - 1)

        acf_values = acf(ts_clean, nlags=nlags)
        pacf_values = pacf(ts_clean, nlags=nlags)

        results = {
            "acf": acf_values.tolist(),
            "pacf": pacf_values.tolist(),
            "significant_acf_lags": [i for i, v in enumerate(acf_values) if abs(v) > 1.96 / np.sqrt(len(ts_clean))],
            "significant_pacf_lags": [i for i, v in enumerate(pacf_values) if abs(v) > 1.96 / np.sqrt(len(ts_clean))],
        }

        self.analysis_results["autocorrelation"] = results
        return results

    def detect_anomalies(self, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in the time series.

        Args:
            method: 'zscore', 'iqr', or 'rolling'
            threshold: Threshold for anomaly detection

        Returns:
            DataFrame with anomaly indicators
        """
        anomalies = pd.DataFrame(index=self.ts.index)
        anomalies["value"] = self.ts
        anomalies["is_anomaly"] = False

        if method == "zscore":
            z_scores = (self.ts - self.ts.mean()) / self.ts.std()
            anomalies["is_anomaly"] = np.abs(z_scores) > threshold
            anomalies["score"] = z_scores

        elif method == "iqr":
            Q1 = self.ts.quantile(0.25)
            Q3 = self.ts.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            anomalies["is_anomaly"] = (self.ts < lower) | (self.ts > upper)
            anomalies["lower_bound"] = lower
            anomalies["upper_bound"] = upper

        elif method == "rolling":
            window = min(30, len(self.ts) // 4)
            rolling_mean = self.ts.rolling(window=window, center=True).mean()
            rolling_std = self.ts.rolling(window=window, center=True).std()
            z_scores = (self.ts - rolling_mean) / rolling_std
            anomalies["is_anomaly"] = np.abs(z_scores) > threshold
            anomalies["score"] = z_scores

        n_anomalies = anomalies["is_anomaly"].sum()
        print(f"🔍 Detected {n_anomalies} anomalies ({n_anomalies/len(self.ts)*100:.2f}%)")

        self.analysis_results["anomalies"] = {
            "method": method,
            "threshold": threshold,
            "n_anomalies": int(n_anomalies),
            "anomaly_percentage": float(n_anomalies / len(self.ts) * 100),
        }

        return anomalies

    def forecast_arima(self, periods: int = 30, order: Optional[Tuple[int, int, int]] = None) -> pd.DataFrame:
        """
        Forecast using ARIMA model.

        Args:
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q) - auto if None

        Returns:
            DataFrame with forecast
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA.")

        # Auto-detect order if not provided
        if order is None:
            order = (1, 1, 1)  # Default order

        # Fit model
        model = ARIMA(self.ts.dropna(), order=order)
        fitted = model.fit()

        # Forecast
        forecast = fitted.get_forecast(steps=periods)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Create future dates
        last_date = self.ts.index[-1]
        if self.freq == "H":
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="H")[1:]
        elif self.freq == "D":
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]
        elif self.freq == "W":
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="W")[1:]
        elif self.freq == "M":
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="ME")[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame(
            {
                "date": future_dates[: len(forecast_mean)],
                "forecast": forecast_mean.values,
                "lower_ci": conf_int.iloc[:, 0].values,
                "upper_ci": conf_int.iloc[:, 1].values,
            }
        )

        self.forecast_results["arima"] = {"order": order, "aic": fitted.aic, "bic": fitted.bic, "forecast": forecast_df}

        return forecast_df

    def forecast_ets(
        self, periods: int = 30, seasonal: Optional[str] = "add", seasonal_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Forecast using Exponential Smoothing (ETS).

        Args:
            periods: Number of periods to forecast
            seasonal: 'add', 'mul', or None
            seasonal_periods: Seasonal period

        Returns:
            DataFrame with forecast
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ETS.")

        # Auto-detect seasonal period
        if seasonal_periods is None:
            if self.freq == "H":
                seasonal_periods = 24
            elif self.freq == "D":
                seasonal_periods = 7
            elif self.freq == "W":
                seasonal_periods = 52
            elif self.freq == "M":
                seasonal_periods = 12
            else:
                seasonal_periods = 4

        # Ensure minimum data length
        if len(self.ts) < 2 * seasonal_periods:
            seasonal = None

        # Fit model
        model = ExponentialSmoothing(
            self.ts.dropna(), seasonal=seasonal, seasonal_periods=seasonal_periods if seasonal else None, trend="add"
        )
        fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(periods)

        # Create future dates
        last_date = self.ts.index[-1]
        if self.freq == "D":
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=self.freq)[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({"date": future_dates[: len(forecast)], "forecast": forecast.values})

        self.forecast_results["ets"] = {
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "aic": fitted.aic,
            "forecast": forecast_df,
        }

        return forecast_df

    def forecast(self, periods: int = 30, method: str = "auto") -> pd.DataFrame:
        """
        Generate forecast using best available method.

        Args:
            periods: Number of periods to forecast
            method: 'auto', 'arima', 'ets', or 'prophet'

        Returns:
            DataFrame with forecast
        """
        if method == "auto":
            # Try multiple methods and pick best
            results = {}

            try:
                results["arima"] = self.forecast_arima(periods)
            except Exception as e:
                print(f"ARIMA failed: {str(e)[:50]}")

            try:
                results["ets"] = self.forecast_ets(periods)
            except Exception as e:
                print(f"ETS failed: {str(e)[:50]}")

            if PROPHET_AVAILABLE:
                try:
                    results["prophet"] = self.forecast_prophet(periods)
                except Exception as e:
                    print(f"Prophet failed: {str(e)[:50]}")

            # Return first successful result
            for name, result in results.items():
                self.best_model_name = name
                return result

            raise ValueError("All forecasting methods failed.")

        elif method == "arima":
            return self.forecast_arima(periods)
        elif method == "ets":
            return self.forecast_ets(periods)
        elif method == "prophet":
            return self.forecast_prophet(periods)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forecast_prophet(self, periods: int = 30) -> pd.DataFrame:
        """
        Forecast using Facebook Prophet.

        Args:
            periods: Number of periods to forecast

        Returns:
            DataFrame with forecast
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        # Prepare data for Prophet
        prophet_data = pd.DataFrame({"ds": self.ts.index, "y": self.ts.values})

        # Fit model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_data)

        # Create future dates
        future = model.make_future_dataframe(periods=periods)

        # Forecast
        forecast = model.predict(future)

        # Get only future predictions
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        forecast_df.columns = ["date", "forecast", "lower_ci", "upper_ci"]

        self.forecast_results["prophet"] = {"forecast": forecast_df}

        return forecast_df

    def analyze(self) -> Dict[str, Any]:
        """
        Run complete time series analysis.

        Returns:
            Dictionary with all analysis results
        """
        print("🔍 Running comprehensive time series analysis...")
        print("=" * 60)

        # Basic stats
        print("📊 Computing basic statistics...")
        self.analysis_results["basic_stats"] = self.get_basic_stats()

        # Stationarity
        print("📈 Testing stationarity...")
        try:
            self.test_stationarity()
        except Exception as e:
            print(f"   ⚠️ Stationarity test failed: {str(e)[:50]}")

        # Decomposition
        print("📉 Decomposing time series...")
        try:
            self.decompose()
        except Exception as e:
            print(f"   ⚠️ Decomposition failed: {str(e)[:50]}")

        # Autocorrelation
        print("🔗 Analyzing autocorrelation...")
        try:
            self.analyze_autocorrelation()
        except Exception as e:
            print(f"   ⚠️ Autocorrelation analysis failed: {str(e)[:50]}")

        # Anomaly detection
        print("🚨 Detecting anomalies...")
        try:
            self.detect_anomalies()
        except Exception as e:
            print(f"   ⚠️ Anomaly detection failed: {str(e)[:50]}")

        print("=" * 60)
        print("✅ Analysis complete!")

        return self.analysis_results

    def plot_time_series(self, figsize: Tuple[int, int] = (14, 6), title: str = "Time Series"):
        """Plot the time series."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.ts.index, self.ts.values, linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel(self.value_column)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_decomposition(self, figsize: Tuple[int, int] = (14, 12)):
        """Plot decomposition components."""
        if self.decomposition is None:
            self.decompose()

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        self.decomposition.observed.plot(ax=axes[0], title="Observed")
        self.decomposition.trend.plot(ax=axes[1], title="Trend")
        self.decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
        self.decomposition.resid.plot(ax=axes[3], title="Residual")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_forecast(self, forecast_df: Optional[pd.DataFrame] = None, figsize: Tuple[int, int] = (14, 6)):
        """Plot forecast with confidence intervals."""
        if forecast_df is None:
            # Use most recent forecast
            for method in ["prophet", "arima", "ets"]:
                if method in self.forecast_results:
                    forecast_df = self.forecast_results[method]["forecast"]
                    break

        if forecast_df is None:
            raise ValueError("No forecast available. Run forecast() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot historical data
        ax.plot(self.ts.index, self.ts.values, label="Historical", linewidth=1)

        # Plot forecast
        ax.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast", color="red", linewidth=2)

        # Plot confidence intervals if available
        if "lower_ci" in forecast_df.columns:
            ax.fill_between(
                forecast_df["date"],
                forecast_df["lower_ci"],
                forecast_df["upper_ci"],
                alpha=0.3,
                color="red",
                label="95% Confidence Interval",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel(self.value_column)
        ax.set_title("Time Series Forecast")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def generate_report(self) -> str:
        """Generate a comprehensive time series analysis report."""
        if not self.analysis_results:
            self.analyze()

        lines = [
            "=" * 70,
            "📊 TIME SERIES ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        # Basic Stats
        if "basic_stats" in self.analysis_results:
            stats = self.analysis_results["basic_stats"]
            lines.extend(
                [
                    "BASIC STATISTICS",
                    "-" * 40,
                    f"  Observations: {stats['count']}",
                    f"  Date Range: {stats['start_date']} to {stats['end_date']}",
                    f"  Frequency: {stats['frequency']}",
                    f"  Mean: {stats['mean']:.2f}",
                    f"  Std Dev: {stats['std']:.2f}",
                    f"  Min: {stats['min']:.2f}",
                    f"  Max: {stats['max']:.2f}",
                    "",
                ]
            )

        # Stationarity
        if "stationarity" in self.analysis_results:
            stat = self.analysis_results["stationarity"]
            lines.extend(
                [
                    "STATIONARITY TESTS",
                    "-" * 40,
                    f"  ADF Test p-value: {stat['adf']['p_value']:.4f}",
                    f"  ADF Stationary: {'Yes' if stat['adf']['is_stationary'] else 'No'}",
                    f"  Conclusion: {stat['conclusion'].upper()}",
                    "",
                ]
            )

        # Decomposition
        if "decomposition" in self.analysis_results:
            decomp = self.analysis_results["decomposition"]
            lines.extend(
                [
                    "DECOMPOSITION",
                    "-" * 40,
                    f"  Model: {decomp['model']}",
                    f"  Period: {decomp['period']}",
                    f"  Trend Strength: {decomp['trend_strength']:.2%}",
                    f"  Seasonal Strength: {decomp['seasonal_strength']:.2%}",
                    "",
                ]
            )

        # Anomalies
        if "anomalies" in self.analysis_results:
            anom = self.analysis_results["anomalies"]
            lines.extend(
                [
                    "ANOMALY DETECTION",
                    "-" * 40,
                    f"  Method: {anom['method']}",
                    f"  Anomalies Found: {anom['n_anomalies']}",
                    f"  Percentage: {anom['anomaly_percentage']:.2f}%",
                    "",
                ]
            )

        lines.extend(["=" * 70, "End of Report", "=" * 70])

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Time Series Analysis Module Demo")
    print("=" * 60)

    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=365, freq="D")

    # Create synthetic time series with trend and seasonality
    trend = np.linspace(100, 150, 365)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonality + noise

    data = pd.DataFrame({"date": dates, "sales": values})

    # Initialize analyzer
    ts = TimeSeriesAnalyzer(data, date_column="date", value_column="sales")

    # Run analysis
    results = ts.analyze()

    # Generate forecast
    print("\n📈 Generating forecast...")
    try:
        forecast = ts.forecast(periods=30)
        print(forecast.head(10))
    except Exception as e:
        print(f"Forecast failed: {e}")

    # Print report
    print("\n" + ts.generate_report())
