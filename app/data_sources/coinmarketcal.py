"""Cliente para CoinMarketCal (requiere API Key/Secret)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests


def fetch_events(max_results: int = 20) -> pd.DataFrame:
    api_key = os.getenv("COINMARKETCAL_API_KEY")
    api_secret = os.getenv("COINMARKETCAL_API_SECRET")
    if not api_key or not api_secret:
        return pd.DataFrame()
    endpoint = "https://pro-api.coinmarketcal.com/v1/events"
    headers = {
        "x-api-key": api_key,
        "x-api-secret": api_secret,
    }
    params = {
        "page": 1,
        "max": max(5, min(max_results, 100)),
        "sort": "created_desc",
    }
    resp = requests.get(endpoint, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    events = resp.json()
    frame = pd.DataFrame(events)
    if frame.empty:
        return frame
    for col in ["date_event", "date_added"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame
