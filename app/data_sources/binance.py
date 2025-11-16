"""Cliente simple para la API pública de Binance (REST)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.binance.com/api/v3"

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


def _request(path: str, params: Dict[str, Any]) -> Any:
    url = f"{BASE_URL}/{path}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Descarga velas OHLCV para un símbolo (sin autenticación)."""
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if start_time is not None:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time is not None:
        params["endTime"] = int(end_time.timestamp() * 1000)
    raw = _request("klines", params)
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = pd.DataFrame(raw, columns=columns)
    if frame.empty:
        return frame
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        frame[col] = frame[col].astype(float)
    frame["number_of_trades"] = frame["number_of_trades"].astype(int)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame["symbol"] = symbol.upper()
    frame["interval"] = interval
    frame = frame.drop(columns=["ignore"])
    return frame


def fetch_order_book(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """Obtiene libro de órdenes (depth) sin clave."""
    raw = _request(
        "depth",
        {
            "symbol": symbol.upper(),
            "limit": min(limit, 5000),
        },
    )
    ts = datetime.now(tz=timezone.utc)
    return {
        "symbol": symbol.upper(),
        "last_update_id": raw.get("lastUpdateId"),
        "timestamp": ts.isoformat(),
        "bids": raw.get("bids", []),
        "asks": raw.get("asks", []),
    }


def fetch_klines_range(
    symbol: str,
    interval: str = "15m",
    days: int = 30,
    batch_limit: int = 1000,
) -> pd.DataFrame:
    """Descarga todas las velas del rango especificado paginando."""
    interval_ms = INTERVAL_TO_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"Intervalo no soportado: {interval}")
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    frames: List[pd.DataFrame] = []
    current_start = start
    while current_start < end:
        current_end = current_start + timedelta(milliseconds=interval_ms * batch_limit)
        frame = fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_limit,
            start_time=current_start,
            end_time=min(current_end, end),
        )
        if frame.empty:
            break
        frames.append(frame)
        last_close = frame["close_time"].max()
        if pd.isna(last_close):
            break
        last_close_dt = last_close.to_pydatetime()
        current_start = last_close_dt + timedelta(milliseconds=interval_ms)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"])
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined
