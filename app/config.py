"""Carga y validación de configuración de la aplicación."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional
import copy

import yaml


@dataclass
class AppSection:
    domain: str
    default_symbol: str
    default_texts: List[str]


@dataclass
class SentimentSection:
    enabled: bool
    provider: str
    task: str
    model_id: str
    device: int
    batch_size: int


@dataclass
class PriceSection:
    enabled: bool
    provider: str
    symbol: str
    model_path: Path
    source: str
    interval: str
    market_symbol: Optional[str]
    lookback_days: int
    fast_ma: int
    slow_ma: int
    rsi_window: int
    feature_columns: List[str]
    model_type: str = "random_forest"
    benchmark_symbol: Optional[str] = None
    huggingface: Optional["HuggingFaceModel"] = None
    asset_family: str = "crypto"


@dataclass
class HuggingFaceModel:
    repo_id: str
    filename: str
    revision: Optional[str] = None
    subfolder: Optional[str] = None
    cache_dir: Optional[str] = None
    token_env_var: Optional[str] = "HF_TOKEN"
    scalers_file: Optional[str] = None
    preproc_file: Optional[str] = None


@dataclass
class Settings:
    app: AppSection
    sentiment: SentimentSection
    price: PriceSection


DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "domain": "generic",
        "default_symbol": "BTC-USD",
        "default_texts": [],
    },
    "sentiment": {
        "enabled": True,
        "provider": "huggingface",
        "task": "sentiment-analysis",
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "device": -1,
        "batch_size": 16,
    },
    "price": {
        "enabled": True,
        "provider": "sklearn",
        "symbol": "BTC-USD",
        "model_path": "models/price-model.joblib",
        "source": "local",  # local | huggingface
        "interval": "1d",
        "market_symbol": None,
        "lookback_days": 120,
        "fast_ma": 7,
        "slow_ma": 21,
        "rsi_window": 14,
        "feature_columns": [
            "return_1d",
            "return_7d",
            "volatility_7d",
            "fast_ma",
            "slow_ma",
            "ma_ratio",
            "rsi",
        ],
        "model_type": "random_forest",
        "benchmark_symbol": None,
        "huggingface": None,
        "asset_family": "crypto",
    },
}


def deep_merge(base: MutableMapping[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina recursivamente dos diccionarios sin mutar los originales."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Settings:
    """Lee un archivo YAML y devuelve una instancia de Settings."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        user_config = yaml.safe_load(fh) or {}

    merged = deep_merge(DEFAULT_CONFIG, user_config)
    base_dir = config_path.parent

    app_section = AppSection(**merged["app"])
    sentiment_section = SentimentSection(**merged["sentiment"])

    price_raw = merged["price"]
    model_path = Path(price_raw["model_path"])
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()
    hf_section = price_raw.get("huggingface")
    hf_config = HuggingFaceModel(**hf_section) if hf_section else None
    price_section = PriceSection(
        **{
            **price_raw,
            "model_path": model_path,
            "huggingface": hf_config,
        }
    )

    return Settings(
        app=app_section,
        sentiment=sentiment_section,
        price=price_section,
    )


def override_symbol(settings: Settings, symbol: str | None) -> Settings:
    """Devuelve una copia del Settings con el símbolo actualizado."""
    if not symbol:
        return settings
    price = replace(settings.price, symbol=symbol)
    app = replace(settings.app, default_symbol=symbol)
    return Settings(app=app, sentiment=settings.sentiment, price=price)
