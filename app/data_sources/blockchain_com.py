"""Datos on-chain públicos desde blockchain.com."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import requests

BASE_URL = "https://api.blockchain.info/charts"


def fetch_chart(chart: str, timespan: str = "30days") -> pd.DataFrame:
    """Descarga una serie pública (p.ej. transactions-per-second) sin clave."""
    url = f"{BASE_URL}/{chart}"
    resp = requests.get(
        url,
        params={"timespan": timespan, "format": "json", "cors": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    payload: Dict[str, Any] = resp.json()
    values = payload.get("values", [])
    frame = pd.DataFrame(values)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["x"], unit="s", utc=True)
    frame = frame.rename(columns={"y": "value"})
    frame["chart"] = chart
    frame["timespan"] = timespan
    frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame[["timestamp", "value", "chart", "timespan", "fetched_at"]]
