import numpy as np
import pandas as pd
from pathlib import Path

from app.config import PriceSection
import app.features.equity_features as equity_features


def sample_candles():
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=40, freq="H"),
            "open": np.linspace(100, 120, 40),
            "high": np.linspace(101, 121, 40),
            "low": np.linspace(99, 119, 40),
            "close": np.linspace(100, 120, 40),
            "volume": np.linspace(1000, 2000, 40),
        }
    )


def test_add_indicators_generates_obv_vwap_ratio():
    indicators = equity_features.add_indicators(sample_candles())
    assert "obv" in indicators.columns
    assert "vwap_ratio" in indicators.columns
    assert indicators["obv"].notna().all()
    assert indicators["vwap_ratio"].notna().all()


def test_build_equity_feature_row(monkeypatch):
    def fake_fetch(symbol, interval, lookback_days):
        return sample_candles()

    def fake_attach(frame, benchmark_symbol, interval, lookback_days):
        frame = frame.copy()
        frame["benchmark_close"] = 100.0
        frame["benchmark_return_1"] = 0.01
        frame["close_to_benchmark"] = 1.0
        return frame

    def fake_corporate(symbol):
        return {
            "days_to_next_earnings": 15.0,
            "has_upcoming_earnings": 1.0,
            "last_dividend": 0.5,
            "dividend_yield": 0.02,
        }

    monkeypatch.setattr(equity_features, "fetch_candles", fake_fetch)
    monkeypatch.setattr(equity_features, "attach_benchmark_features", fake_attach)
    monkeypatch.setattr(equity_features, "fetch_corporate_features", fake_corporate)

    price = PriceSection(
        enabled=True,
        provider="local_sklearn",
        symbol="TSLA",
        model_path=Path("models/local_price_TSLA_equity_1h.joblib"),
        source="local",
        interval="1h",
        market_symbol="TSLA",
        lookback_days=30,
        fast_ma=7,
        slow_ma=21,
        rsi_window=14,
        feature_columns=[
            "benchmark_close",
            "benchmark_return_1",
            "close_to_benchmark",
            "days_to_next_earnings",
            "dividend_yield",
        ],
        benchmark_symbol="^GSPC",
        asset_family="equity",
    )

    row = equity_features.build_equity_feature_row(price, price.feature_columns)
    for col in price.feature_columns:
        assert col in row.columns
