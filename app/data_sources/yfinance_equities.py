"""Utilidades para descargar OHLCV de acciones vía yfinance."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

FMP_API_KEY = os.environ.get("FMP_API_KEY")
FMP_EARNINGS_URL = "https://financialmodelingprep.com/api/v3/earning_calendar"


def fetch_equity_history(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Descarga historial OHLCV de un ticker (TSLA, NVDA, etc.)."""
    if days <= 0:
        days = 30
    period = f"{days}d"
    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()
    frame = data.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        flat_cols = []
        for col in frame.columns:
            if isinstance(col, tuple):
                base = next((str(level) for level in col if level), "")
            else:
                base = str(col)
            flat_cols.append(base)
        frame.columns = flat_cols
    time_col = "Datetime" if "Datetime" in frame.columns else "Date"
    frame["open_time"] = pd.to_datetime(frame[time_col], utc=True)
    frame["close_time"] = frame["open_time"]
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    needed = ["open", "high", "low", "close", "volume", "open_time", "close_time"]
    for col in needed:
        if col not in frame.columns:
            frame[col] = 0.0 if col != "open_time" and col != "close_time" else datetime.now(tz=timezone.utc)
    return frame[needed + [c for c in frame.columns if c not in needed]]


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (volume.fillna(0.0) * direction).cumsum()


def compute_vwap(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].replace(0, np.nan)
    tp_vol = typical_price * volume
    rolling_vwap = tp_vol.rolling(window).sum() / volume.rolling(window).sum()
    return rolling_vwap.fillna(method="bfill").fillna(method="ffill")


def attach_benchmark_features(
    frame: pd.DataFrame, benchmark_symbol: Optional[str], interval: str, lookback_days: int
) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [col[0] if isinstance(col, tuple) else col for col in frame.columns]
    if not benchmark_symbol:
        frame["benchmark_close"] = 0.0
        frame["benchmark_return_1"] = 0.0
        frame["close_to_benchmark"] = 1.0
        return frame
    bench = fetch_equity_history(benchmark_symbol, interval=interval, days=lookback_days)
    if bench.empty:
        frame["benchmark_close"] = 0.0
        frame["benchmark_return_1"] = 0.0
        frame["close_to_benchmark"] = 1.0
        return frame
    bench = bench[["close_time", "close"]].rename(columns={"close": "benchmark_close"})
    bench = bench.sort_values("close_time")
    bench["benchmark_return_1"] = bench["benchmark_close"].pct_change().fillna(0.0)
    bench["close_to_benchmark"] = bench["benchmark_close"]
    merged = pd.merge_asof(
        frame.sort_values("close_time"),
        bench[["close_time", "benchmark_close", "benchmark_return_1"]],
        on="close_time",
        direction="nearest",
    )
    merged["benchmark_close"] = merged["benchmark_close"].ffill().bfill().fillna(0.0)
    merged["benchmark_return_1"] = merged["benchmark_return_1"].fillna(0.0)
    merged["close_to_benchmark"] = merged["close"] / (merged["benchmark_close"] + 1e-9)
    return merged


def _fetch_fmp_earnings(symbol: str) -> Optional[Dict[str, float]]:
    """Consulta FinancialModelingPrep en busca del próximo earnings."""
    if not FMP_API_KEY:
        return None
    params = {
        "symbol": symbol,
        "limit": 16,
        "apikey": FMP_API_KEY,
    }
    try:
        resp = requests.get(FMP_EARNINGS_URL, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:  # pragma: no cover - red de terceros
        return None
    try:
        data = resp.json()
    except ValueError:  # pragma: no cover
        return None
    if not isinstance(data, list):
        return None
    now = datetime.now(timezone.utc).date()
    next_date: Optional[datetime] = None
    for entry in data:
        date_str = entry.get("date")
        if not date_str:
            continue
        try:
            event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if event_date < now:
            continue
        if next_date is None or event_date < next_date:
            next_date = event_date
    if not next_date:
        return None
    delta_days = (next_date - now).days
    return {
        "days_to_next_earnings": float(delta_days),
        "has_upcoming_earnings": 1.0 if delta_days >= 0 else 0.0,
    }


def fetch_corporate_features(symbol: str) -> Dict[str, float]:
    base_symbol = symbol.replace("-USD", "")
    features: Dict[str, float] = {
        "days_to_next_earnings": 0.0,
        "has_upcoming_earnings": 0.0,
        "dividend_yield": 0.0,
        "last_dividend": 0.0,
    }
    try:
        ticker = yf.Ticker(base_symbol)
    except Exception:  # pragma: no cover
        return features
    # Earnings calendar
    try:
        cal = ticker.get_calendar() if hasattr(ticker, "get_calendar") else getattr(ticker, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            row = cal.T.get("Earnings Date")
            if row is not None:
                dates = pd.to_datetime(row.dropna(), errors="coerce")
                dates = dates[dates >= datetime.now()]
                if not dates.empty:
                    next_date = dates.min()
                    delta = (next_date - datetime.now()).days
                    features["days_to_next_earnings"] = float(delta)
                    features["has_upcoming_earnings"] = 1.0 if delta >= 0 else 0.0
    except Exception:  # pragma: no cover
        pass
    # Fallback a FinancialModelingPrep si no hay earnings desde Yahoo
    if features["days_to_next_earnings"] == 0.0:
        fallback = _fetch_fmp_earnings(base_symbol)
        if fallback:
            features.update(fallback)
    # Dividend info
    try:
        dividends = ticker.dividends
        if isinstance(dividends, pd.Series) and not dividends.empty:
            last_div = float(dividends.iloc[-1])
            features["last_dividend"] = last_div
            hist = ticker.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                if price:
                    features["dividend_yield"] = (last_div * 4) / price
    except Exception:  # pragma: no cover
        pass
    return features


def fetch_macro_series(symbol: str, interval: str, days: int) -> pd.DataFrame:
    period = f"{max(days, 5)}d"
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat.append(next((str(level) for level in col if level), ""))
            else:
                flat.append(str(col))
        df.columns = flat
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["close_time"] = pd.to_datetime(df[time_col], utc=True)
    rename_map = {
        "Close": f"macro_{symbol}_close",
        "Adj Close": f"macro_{symbol}_adj_close",
        "Volume": f"macro_{symbol}_volume",
    }
    cols = ["close_time"] + [col for col in rename_map if col in df.columns]
    return df[cols].rename(columns=rename_map)
