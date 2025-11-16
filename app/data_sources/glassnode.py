"""Cliente para la API de Glassnode."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests


def fetch_metric(metric: str, asset: str = "BTC", since_days: int = 30) -> pd.DataFrame:
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        return pd.DataFrame()
    endpoint = f"https://api.glassnode.com/v1/metrics/{metric}"
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=since_days)
    params = {
        "a": asset.upper(),
        "s": int(start.timestamp()),
        "u": int(end.timestamp()),
        "i": "1d",
        "api_key": api_key,
    }
    resp = requests.get(endpoint, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["t"] = pd.to_datetime(frame["t"], unit="s", utc=True)
    frame = frame.rename(columns={"t": "timestamp", "v": metric.replace("/", "_")})
    return frame
