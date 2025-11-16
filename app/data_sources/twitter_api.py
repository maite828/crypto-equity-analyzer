"""Cliente simple para Twitter (X) usando la API v2."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import requests


def fetch_recent_tweets(query: str, max_results: int = 20, bearer_token: Optional[str] = None) -> pd.DataFrame:
    token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return pd.DataFrame()
    endpoint = "https://api.twitter.com/2/tweets/search/recent"
    params: Dict[str, str] = {
        "query": query,
        "max_results": str(max(10, min(max_results, 100))),
        "tweet.fields": "created_at,author_id,lang",
    }
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(endpoint, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame
