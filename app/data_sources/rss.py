"""Parsers de RSS públicos (CoinDesk, etc.)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import feedparser


def fetch_feed(url: str) -> pd.DataFrame:
    """Descarga un feed RSS y lo devuelve como DataFrame."""
    parsed = feedparser.parse(url)
    entries = parsed.entries or []
    records = []
    for entry in entries:
        published = entry.get("published_parsed")
        published_dt = (
            datetime(*published[:6], tzinfo=timezone.utc) if published else None
        )
        records.append(
            {
                "title": entry.get("title"),
                "summary": entry.get("summary"),
                "link": entry.get("link"),
                "published": published_dt,
                "feed_title": parsed.feed.get("title", ""),
                "feed_url": url,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame
