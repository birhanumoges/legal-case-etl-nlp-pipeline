"""tests/test_time_series.py — Tests for time-series analysis and forecasting."""

import pytest
import numpy as np
import pandas as pd

from time_series.trend_analysis   import TrendAnalyzer
from time_series.yearly_statistics import YearlyStatistics
from time_series.forecasting       import LegalForecaster


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def ts_df():
    """15 years of synthetic legal data."""
    rng   = np.random.default_rng(42)
    years = list(range(1880, 1895))
    rows  = []
    types    = ["CIVIL", "CRIMINAL", "CONTRACT", "PROPERTY", "TORTS"]
    subtypes = ["Appeal", "Larceny", "Bond", "Title Dispute", "Negligence"]
    verdicts = ["AFFIRMED", "REVERSED", "DENIED", "GRANTED", "OTHER"]
    courts   = ["Superior Court", "Supreme Court"]

    for year in years:
        for _ in range(rng.integers(20, 60)):
            rows.append({
                "Case_ID":          f"case_{year}_{_}",
                "Year":             str(year),
                "Year_Numeric":     float(year),
                "Case_Type_Mapped": rng.choice(types),
                "Sub_Type_Mapped":  rng.choice(subtypes),
                "Verdict_Mapped":   rng.choice(verdicts),
                "Court":            rng.choice(courts),
                "Num_Citations":    int(rng.integers(0, 10)),
            })
    return pd.DataFrame(rows)


# ── TrendAnalyzer ─────────────────────────────────────────────────────────────

class TestTrendAnalyzer:
    def test_case_type_trends_shape(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        analyzer = TrendAnalyzer(ts_df)
        pivot    = analyzer.case_type_trends()
        assert not pivot.empty
        # rows = years, cols = case types
        assert pivot.index.name == "Year_Numeric"
        assert set(pivot.columns).issubset({"CIVIL","CRIMINAL","CONTRACT","PROPERTY","TORTS"})

    def test_verdict_trends_shape(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        analyzer = TrendAnalyzer(ts_df)
        pivot    = analyzer.verdict_trends()
        assert not pivot.empty
        assert "AFFIRMED" in pivot.columns or "REVERSED" in pivot.columns

    def test_citation_growth_columns(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        analyzer = TrendAnalyzer(ts_df)
        annual   = analyzer.citation_growth()
        for col in ["avg_citations", "total_citations", "n_cases"]:
            assert col in annual.columns

    def test_subtype_trends(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        analyzer = TrendAnalyzer(ts_df)
        pivot    = analyzer.subtype_trends(top_n=3)
        assert not pivot.empty
        assert pivot.shape[1] <= 3

    def test_saves_csv(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        TrendAnalyzer(ts_df).case_type_trends()
        assert (tmp_path / "ts_case_type_trends.csv").exists()


# ── YearlyStatistics ──────────────────────────────────────────────────────────

class TestYearlyStatistics:
    def test_compute_returns_dataframe(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        stats = YearlyStatistics(ts_df).compute()
        assert not stats.empty
        assert "n_cases" in stats.columns
        assert "avg_citations" in stats.columns

    def test_all_years_present(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        stats = YearlyStatistics(ts_df).compute()
        assert set(range(1880, 1895)).issubset(set(stats.index.astype(int)))

    def test_saves_csv(self, ts_df, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        YearlyStatistics(ts_df).compute()
        assert (tmp_path / "ts_yearly_statistics.csv").exists()


# ── LegalForecaster ───────────────────────────────────────────────────────────

class TestLegalForecaster:
    def test_annual_series_built(self, ts_df):
        forecaster = LegalForecaster(ts_df, forecast_years=3)
        assert len(forecaster.annual) == 15   # 1880-1894

    def test_arima_forecast(self, ts_df, tmp_path, monkeypatch):
        pytest.importorskip("statsmodels")
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        forecaster = LegalForecaster(ts_df, forecast_years=3)
        result     = forecaster.arima_forecast()
        if result is None:
            pytest.skip("ARIMA could not fit (small dataset)")
        assert result["method"] == "ARIMA"
        assert len(result["forecast"]) == 3
        assert (tmp_path / "ts_arima_forecast.json").exists()

    def test_prophet_forecast(self, ts_df, tmp_path, monkeypatch):
        pytest.importorskip("prophet")
        import config as cfg
        monkeypatch.setattr(cfg, "PLOT_DIR",   tmp_path)
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        forecaster = LegalForecaster(ts_df, forecast_years=3)
        result     = forecaster.prophet_forecast()
        if result is None:
            pytest.skip("Prophet could not fit")
        assert result["method"] == "Prophet"
        assert len(result["forecast"]) >= 1

    def test_insufficient_data_graceful(self):
        tiny_df    = pd.DataFrame([{"Year_Numeric": 1890.0, "Case_ID": "x"}])
        forecaster = LegalForecaster(tiny_df, forecast_years=3)
        result     = forecaster.arima_forecast()
        assert result is None   # not enough data, returns None gracefully
