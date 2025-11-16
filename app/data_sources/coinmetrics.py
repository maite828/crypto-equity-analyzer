"""Cliente básico para CoinMetrics community API."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests


def fetch_asset_metrics(
    assets: List[str], metrics: List[str], since_days: int = 30, frequency: str = "1d"
) -> pd.DataFrame:
    api_key = os.getenv("COINMETRICS_API_KEY")
    if not api_key:
        return pd.DataFrame()
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=since_days)
    params = {
        "assets": ",".join(assets),
        "metrics": ",".join(metrics),
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "frequency": frequency,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame
