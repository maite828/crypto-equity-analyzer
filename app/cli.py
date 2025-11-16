"""CLI para ejecutar análisis de sentimiento y precios de forma configurable."""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Sequence

import pandas as pd

from .config import Settings, load_config, override_symbol
from .pricing import PriceService
from .sentiment import SentimentService

LOGGER = logging.getLogger("app.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza textos y series de precios usando modelos configurables."
    )
    parser.add_argument(
        "--config",
        default="configs/sample-config.yaml",
        help="Ruta al archivo YAML de configuración.",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="Textos a analizar (si se omite se usan los del archivo de config).",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Override del símbolo configurado (por ejemplo BTC-USD).",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="No ejecutar la parte de sentimiento.",
    )
    parser.add_argument(
        "--skip-price",
        action="store_true",
        help="No ejecutar la parte de precios.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Nivel de logging (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def _gather_texts(args: argparse.Namespace, config: Settings) -> List[str]:
    env_texts = os.getenv("TEXTS")
    texts: List[str] = []
    if args.texts:
        texts.extend(args.texts)
    if env_texts:
        texts.extend(t.strip() for t in env_texts.split("||") if t.strip())
    if not texts:
        texts.extend(config.app.default_texts)
    # Elimina duplicados conservando orden
    seen = set()
    ordered: List[str] = []
    for text in texts:
        if text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered


def _print_dataframe(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False))


def _print_series(title: str, series: pd.Series) -> None:
    print(f"\n{title}")
    print(series.to_string())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        settings = load_config(args.config)
    except FileNotFoundError as exc:
        LOGGER.error(exc)
        raise SystemExit(1) from exc

    symbol_override = args.symbol or os.getenv("SYMBOL")
    settings = override_symbol(settings, symbol_override)

    texts = _gather_texts(args, settings)

    if not args.skip_sentiment and settings.sentiment.enabled:
        try:
            sentiment_service = SentimentService(settings.sentiment)
            if texts:
                sentiment_df = sentiment_service.analyze(texts)
                _print_dataframe("Resultados de sentimiento:", sentiment_df)
            else:
                LOGGER.info("No se proporcionaron textos para analizar.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Fallo al ejecutar el módulo de sentimiento: %s", exc)

    if not args.skip_price and settings.price.enabled:
        try:
            price_service = PriceService(settings.price)
            result = price_service.predict(settings.price.symbol)
            _print_series(f"Predicción de precios para {settings.price.symbol}:", result)
        except FileNotFoundError as exc:
            LOGGER.warning(exc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Fallo al ejecutar el módulo de precios: %s", exc)


if __name__ == "__main__":
    main()
