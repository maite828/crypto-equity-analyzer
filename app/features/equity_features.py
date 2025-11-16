"""Construye features para acciones usando yfinance."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import PriceSection
from app.data_sources.yfinance_equities import (
    attach_benchmark_features,
    compute_obv,
    compute_vwap,
    fetch_corporate_features,
)

LOGGER = logging.getLogger(__name__)


def fetch_candles(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    period = f"{max(lookback_days, 5)}d"
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    frame = data.reset_index()
    time_col = "Datetime" if "Datetime" in frame.columns else "Date"
    frame["open_time"] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return frame[["open_time", "open", "high", "low", "close", "volume"]]


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values("open_time").reset_index(drop=True).copy()
    df["mid_price"] = (df["open"] + df["close"]) / 2
    df["return_1"] = df["close"].pct_change().fillna(0)
    df["return_4"] = df["close"].pct_change(4).fillna(0)
    df["return_16"] = df["close"].pct_change(16).fillna(0)
    df["volatility_16"] = df["return_1"].rolling(16).std().fillna(0)
    df["volatility_64"] = df["return_1"].rolling(64).std().fillna(0)
    df["ema_fast"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_ratio"] = df["ema_fast"] / (df["ema_slow"] + 1e-9)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["obv"] = compute_obv(df["close"], df.get("volume", pd.Series(0, index=df.index)))
    vwap = compute_vwap(df, 20)
    df["vwap_20"] = vwap
    df["vwap_ratio"] = df["close"] / (vwap + 1e-9)
    return df


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_equity_feature_row(config: PriceSection, feature_columns: List[str]) -> pd.DataFrame:
    candles = fetch_candles(config.market_symbol or config.symbol, config.interval, config.lookback_days)
    if candles.empty:
        raise RuntimeError("No se pudieron obtener velas de yfinance para el ticker indicado.")
    feats = add_indicators(candles)
    feats["close_time"] = feats["open_time"]
    feats = attach_benchmark_features(
        feats,
        config.benchmark_symbol,
        config.interval,
        config.lookback_days,
    )
    corp = fetch_corporate_features(config.symbol)
    for key, value in corp.items():
        feats[key] = value
    latest = feats.iloc[[-1]].copy()
    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = 0.0
    return latest[feature_columns]
