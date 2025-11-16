"""Construye el vector de características para modelos locales (sklearn)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import PriceSection
from app.data_sources.binance import fetch_klines, fetch_klines_range, fetch_order_book
from app.data_sources.coingecko import fetch_market_overview
from app.data_sources.blockchain_com import fetch_chart
from app.assets import COINGECKO_MAP

LOGGER = logging.getLogger(__name__)

def to_market_symbol(symbol: str) -> str:
    candidate = symbol.replace("-", "").replace("/", "").upper()
    if candidate.endswith("USD") and not candidate.endswith("USDT"):
        candidate = f"{candidate}T"
    return candidate


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.sort_values("open_time").reset_index(drop=True)
    frame["mid_price"] = (frame["open"] + frame["close"]) / 2
    frame["return_1"] = frame["close"].pct_change().fillna(0)
    frame["return_4"] = frame["close"].pct_change(4).fillna(0)
    frame["return_16"] = frame["close"].pct_change(16).fillna(0)
    frame["volatility_16"] = frame["return_1"].rolling(16).std().fillna(0)
    frame["volatility_64"] = frame["return_1"].rolling(64).std().fillna(0)
    frame["ema_fast"] = frame["close"].ewm(span=8, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=21, adjust=False).mean()
    frame["ema_ratio"] = frame["ema_fast"] / (frame["ema_slow"] + 1e-9)
    frame["rsi_14"] = compute_rsi(frame["close"], 14)
    frame["atr_14"] = compute_atr(frame, 14)
    frame["close_time"] = frame["close_time"].dt.round("1s")
    return frame


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift()).abs()
    low_close_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def orderbook_features(symbol: str, depth_limit: int = 100) -> Dict[str, float]:
    snapshot = fetch_order_book(symbol, limit=depth_limit)
    bids = pd.DataFrame(snapshot.get("bids", []), columns=["price", "quantity"])
    asks = pd.DataFrame(snapshot.get("asks", []), columns=["price", "quantity"])
    bids["price"] = bids["price"].astype(float)
    bids["quantity"] = bids["quantity"].astype(float)
    asks["price"] = asks["price"].astype(float)
    asks["quantity"] = asks["quantity"].astype(float)
    bid_vol = bids["quantity"].sum()
    ask_vol = asks["quantity"].sum()
    spread = asks["price"].min() - bids["price"].max()
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    pressure = imbalance * (bid_vol + ask_vol)
    return {
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread": spread,
        "ob_imbalance": imbalance,
        "ob_pressure": pressure,
    }


def fetch_spot_metrics(symbol: str) -> Dict[str, float]:
    coin_id = COINGECKO_MAP.get(symbol)
    if not coin_id:
        return {}
    frame = fetch_market_overview([coin_id])
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {
        "current_price": row.get("current_price", 0.0),
        "market_cap": row.get("market_cap", 0.0),
        "market_cap_change": 0.0,
        "total_volume": row.get("total_volume", 0.0),
        "price_change_percentage_24h": row.get("price_change_percentage_24h", 0.0),
        "price_change_percentage_7d_in_currency": row.get("price_change_percentage_7d_in_currency", 0.0),
    }


def fetch_onchain_metrics(timespan: str = "7days") -> Dict[str, float]:
    metrics = {}
    for metric in ["transactions-per-second", "market-price"]:
        try:
            chart = fetch_chart(metric, timespan=timespan)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("No se pudo descargar métrica %s: %s", metric, exc)
            continue
        if chart.empty:
            continue
        metrics[f"onchain_{metric}"] = float(chart["value"].iloc[-1])
    return metrics


def build_feature_row(
    config: PriceSection,
    feature_columns: List[str],
    depth_limit: int = 100,
) -> pd.DataFrame:
    market_symbol = to_market_symbol(config.market_symbol or config.symbol)
    candles = fetch_klines_range(
        market_symbol,
        interval=config.interval,
        days=1,
        batch_limit=1000,
    )
    if candles.empty:
        candles = fetch_klines(market_symbol, interval=config.interval, limit=200)
    if candles.empty:
        raise RuntimeError("No se pudieron obtener velas recientes para construir features.")
    features_df = add_technical_indicators(candles)
    latest = features_df.iloc[[-1]].copy()

    try:
        ob_feats = orderbook_features(market_symbol, depth_limit=depth_limit)
        for key, value in ob_feats.items():
            latest[key] = value
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("No se pudo obtener order book: %s", exc)

    try:
        spot_feats = fetch_spot_metrics(market_symbol)
        for key, value in spot_feats.items():
            latest[key] = value
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("No se pudo obtener métricas spot: %s", exc)

    onchain_feats = fetch_onchain_metrics()
    for key, value in onchain_feats.items():
        latest[key] = value

    # Asegurar que todas las columnas existen
    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = 0.0

    return latest[feature_columns]
