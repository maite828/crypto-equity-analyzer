"""Carga el inventario de activos cripto desde assets/crypto_assets.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ruamel.yaml import YAML

ASSETS_FILE = Path(__file__).resolve().parents[2] / "assets" / "crypto_assets.yaml"
yaml = YAML(typ="safe")
CRYPTO_ASSETS: List[Dict[str, Any]] = yaml.load(ASSETS_FILE.read_text(encoding="utf-8")) or []


def _enabled(entry: Dict[str, Any]) -> bool:
    return bool(entry.get("enabled", False))


def _ranking_enabled(entry: Dict[str, Any]) -> bool:
    return bool(entry.get("ranking_enabled", False))


ALL_CRYPTO_SYMBOLS = [entry["symbol"] for entry in CRYPTO_ASSETS]
DEFAULT_CRYPTO_SYMBOLS = [entry["symbol"] for entry in CRYPTO_ASSETS if _enabled(entry)]
DEFAULT_COIN_IDS = [entry["coin_id"] for entry in CRYPTO_ASSETS if _enabled(entry)]
DEFAULT_COINGECKO_IDS = DEFAULT_COIN_IDS
COINGECKO_MAP = {entry["symbol"]: entry["coin_id"] for entry in CRYPTO_ASSETS}
CRYPTO_RANKING_SYMBOLS = [
    entry.get("ranking_symbol", f'{entry["ticker"]}-USD')
    for entry in CRYPTO_ASSETS
    if _ranking_enabled(entry)
]
CRYPTO_TICKERS = [entry["ticker"] for entry in CRYPTO_ASSETS]
TICKER_TO_SYMBOL = {entry["ticker"]: entry["symbol"] for entry in CRYPTO_ASSETS}
SYMBOL_TO_TICKER = {entry["symbol"]: entry["ticker"] for entry in CRYPTO_ASSETS}
