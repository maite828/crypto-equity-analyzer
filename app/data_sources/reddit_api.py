"""Descarga posts recientes de subreddits públicos."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests


def fetch_subreddit_posts(subreddit: str, limit: int = 25) -> pd.DataFrame:
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    params = {"limit": max(1, min(limit, 100))}
    headers = {"User-Agent": "cryptoprediction-bot/0.1"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    children = payload.get("data", {}).get("children", [])
    records = []
    for child in children:
        data = child.get("data", {})
        records.append(
            {
                "id": data.get("id"),
                "title": data.get("title"),
                "author": data.get("author"),
                "score": data.get("score"),
                "subreddit": subreddit,
                "created_utc": pd.to_datetime(data.get("created_utc"), unit="s", utc=True),
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame
