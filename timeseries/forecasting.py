"""time_series/forecasting.py — ARIMA + Prophet legal volume forecasting."""

from __future__ import annotations
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import PLOT_DIR, REPORT_DIR, FORECAST_YEARS
from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class LegalForecaster:
    """
    Forecast total annual case volumes using ARIMA and Prophet.
    Falls back gracefully if libraries not installed.
    """

    def __init__(self, df: pd.DataFrame, forecast_years: int = FORECAST_YEARS):
        self.forecast_years = forecast_years
        self.annual = self._build_annual(df)

    @staticmethod
    def _build_annual(df: pd.DataFrame) -> pd.Series:
        if "Year_Numeric" not in df.columns:
            return pd.Series(dtype=float)
        return (
            df[df["Year_Numeric"].notna()]
            .groupby("Year_Numeric")
            .size()
            .astype(float)
        )

    # ── ARIMA ────────────────────────────────────────────────────────
    def arima_forecast(self) -> dict | None:
        if len(self.annual) < 10:
            logger.warning("Not enough yearly data for ARIMA (need ≥10 years).")
            return None
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            logger.warning("statsmodels not installed; ARIMA skipped.")
            return None

        y = self.annual.values
        best_aic, best_order, best_model = np.inf, (1, 1, 1), None
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        m = ARIMA(y, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order, best_model = m.aic, (p, d, q), m
                    except Exception:
                        pass

        if best_model is None:
            return None

        forecast_res = best_model.forecast(steps=self.forecast_years)
        last_year    = int(self.annual.index[-1])
        future_years = list(range(last_year + 1, last_year + self.forecast_years + 1))
        result = {
            "method": "ARIMA",
            "order": best_order,
            "aic": round(float(best_aic), 2),
            "forecast": dict(zip(map(str, future_years), forecast_res.tolist())),
        }
        self._plot_forecast(self.annual, future_years, forecast_res, "ARIMA")
        path = REPORT_DIR / "ts_arima_forecast.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("ARIMA forecast saved → %s", path)
        return result

    # ── Prophet ──────────────────────────────────────────────────────
    def prophet_forecast(self) -> dict | None:
        if len(self.annual) < 5:
            return None
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("prophet not installed; Prophet forecast skipped.")
            return None

        df_p = pd.DataFrame({
            "ds": pd.to_datetime([f"{int(y)}-01-01" for y in self.annual.index]),
            "y":  self.annual.values,
        })
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.80)
        m.fit(df_p)
        future  = m.make_future_dataframe(periods=self.forecast_years, freq="YS")
        forecast = m.predict(future)
        fc_rows  = forecast[forecast["ds"].dt.year > int(self.annual.index[-1])]

        result = {
            "method": "Prophet",
            "forecast": {
                str(int(r["ds"].year)): round(float(r["yhat"]), 1)
                for _, r in fc_rows.iterrows()
            },
        }
        fig = m.plot(forecast)
        fig.suptitle("Prophet Forecast — Annual Legal Cases", fontsize=12)
        fig.savefig(PLOT_DIR / "ts_prophet_forecast.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        path = REPORT_DIR / "ts_prophet_forecast.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Prophet forecast saved → %s", path)
        return result

    # ── combined run ─────────────────────────────────────────────────
    def run_all(self):
        logger.info("Running forecasting …")
        a = self.arima_forecast()
        p = self.prophet_forecast()
        return {"arima": a, "prophet": p}

    # ── plot helper ──────────────────────────────────────────────────
    def _plot_forecast(self, history: pd.Series, future_years: list,
                       forecast: np.ndarray, method: str):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history.index, history.values, "b-o", label="Historical", markersize=3)
        ax.plot(future_years, forecast, "r--o", label=f"{method} Forecast", markersize=4)
        ax.axvline(x=history.index[-1], color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"Legal Case Volume Forecast — {method}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Annual Cases")
        ax.legend()
        plt.tight_layout()
        fname = f"ts_{method.lower()}_forecast.png"
        fig.savefig(PLOT_DIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Forecast plot → %s", PLOT_DIR / fname)
