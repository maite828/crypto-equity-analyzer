"""Cliente para la API pública de CoinGecko."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.coingecko.com/api/v3"


def _request(path: str, params: Dict[str, Any]) -> Any:
    url = f"{BASE_URL}/{path}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_market_overview(
    ids: Iterable[str],
    vs_currency: str = "usd",
) -> pd.DataFrame:
    """Descarga métricas de mercado para una lista de activos."""
    coin_ids = ",".join(ids)
    data = _request(
        "coins/markets",
        {
            "vs_currency": vs_currency,
            "ids": coin_ids,
            "price_change_percentage": "1h,24h,7d",
            "order": "market_cap_desc",
            "per_page": len(ids),
            "page": 1,
            "sparkline": "false",
        },
    )
    frame = pd.DataFrame(data)
    frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame


def fetch_status_updates(category: Optional[str] = None, project_type: Optional[str] = None) -> pd.DataFrame:
    """Obtiene actualizaciones/noticias públicas que CoinGecko agrega."""
    params: Dict[str, Any] = {"per_page": 100, "page": 1}
    if category:
        params["category"] = category
    if project_type:
        params["project_type"] = project_type
    try:
        payload = _request("status_updates", params)
    except requests.HTTPError as exc:
        # Endpoint a veces devuelve 404 cuando no hay novedades recientes.
        if exc.response is not None and exc.response.status_code == 404:
            return pd.DataFrame()
        raise
    updates = payload.get("status_updates", [])
    frame = pd.DataFrame(updates)
    if not frame.empty:
        frame["fetched_at"] = datetime.now(tz=timezone.utc)
    return frame
