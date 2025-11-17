#!/usr/bin/env python3
"""Ranking multi-símbolo para acciones usando modelos locales."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.pricing import PriceService
from app.sentiment import SentimentService
from app.features.equity_features import build_equity_feature_row

DEFAULT_EQUITY_SYMBOLS = ["TSLA", "NIO", "NVDA", "AMZN", "GOOGL"]


def fetch_close(symbol: str) -> Optional[float]:
    hist = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=False)
    if hist.empty:
        return None
    value = hist["Close"].iloc[-1]
    return float(value if not hasattr(value, "iloc") else value.iloc[0])


def fetch_headlines(symbol: str, limit: int) -> List[str]:
    try:
        ticker = yf.Ticker(symbol.replace("-USD", ""))
        news = ticker.news
    except Exception:  # pragma: no cover
        return []
    if not news:
        return []
    titles: List[str] = []
    for item in news:
        title = item.get("title")
        if title:
            titles.append(title.strip())
        if len(titles) >= limit:
            break
    return titles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ranking multi-símbolo para acciones (modelos locales).")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_EQUITY_SYMBOLS, help="Tickers a evaluar.")
    parser.add_argument("--interval", default="1h", help="Intervalo del modelo local (por defecto 1h).")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directorio donde se encuentran los YAML local-model-*.yaml.",
    )
    parser.add_argument("--output", default=None, help="Ruta CSV para guardar el ranking.")
    parser.add_argument("--sentiment-headlines", type=int, default=5, help="Noticias recientes a analizar por símbolo.")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging.")
    return parser.parse_args()


def build_record(config_path: Path, headlines_limit: int) -> Optional[dict]:
    settings = load_config(config_path)
    price_service = PriceService(settings.price)
    sentiment_service = SentimentService(settings.sentiment) if settings.sentiment.enabled else None
    prediction = price_service.predict(settings.price.symbol)
    predicted_price = float(
        prediction.get("predicted_price")
        or prediction.get("expected_price")
        or prediction.get("price_t_plus_1")
        or prediction.get("price_t_plus_7")
    )
    current_price = fetch_close(settings.price.symbol)
    if current_price is None:
        logging.warning("No se pudo obtener precio actual para %s", settings.price.symbol)
        return None
    delta = predicted_price - current_price
    delta_pct = (delta / current_price) * 100 if current_price else 0.0
    sentiment_label = None
    sentiment_confidence = None
    if sentiment_service:
        texts = fetch_headlines(settings.price.symbol, headlines_limit)
        if not texts:
            texts = settings.app.default_texts or []
        if texts:
            sentiment_df = sentiment_service.analyze(texts)
            if not sentiment_df.empty:
                sentiment_label = sentiment_df["sentiment"].mode().iloc[0]
                sentiment_confidence = float(sentiment_df["confidence"].mean())

    context = build_equity_feature_row(settings.price, None).iloc[0]
    return {
        "asset": settings.price.symbol,
        "current_price": current_price,
        "expected_price_1d": predicted_price,
        "delta_1d": delta,
        "delta_1d_pct": delta_pct,
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_confidence,
        "benchmark_return_1": context.get("benchmark_return_1"),
        "days_to_next_earnings": context.get("days_to_next_earnings"),
        "has_upcoming_earnings": context.get("has_upcoming_earnings"),
    }


def summarize(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    best = df.iloc[0]
    delta = best["delta_1d"]
    label = best.get("sentiment")
    conf = best.get("sentiment_confidence")
    sentiment_str = ""
    if label:
        sentiment_str = f" Sentimiento dominante: {label}"
        if conf:
            sentiment_str += f" ({conf:.2f})."
        else:
            sentiment_str += "."
    if delta > 0:
        msg = (
            f"Ranking equity: {best['asset']} lidera con una variación esperada de "
            f"{delta:+.2f} USD ({best['delta_1d_pct']:+.2f}%).{sentiment_str}"
        )
    else:
        msg = (
            f"Ranking equity: {best['asset']} lidera pero con delta negativo {delta:+.2f} USD."
            " Considera esperar una señal alcista."
        )
    return msg


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    records: List[dict] = []
    for symbol in args.symbols:
        config_path = Path(args.config_dir) / f"local-model-{symbol.lower()}-{args.interval}.yaml"
        if not config_path.exists():
            logging.warning("Config %s no encontrada, omitiendo %s.", config_path, symbol)
            continue
        record = build_record(config_path, args.sentiment_headlines)
        if record:
            records.append(record)
    if not records:
        logging.warning("No se generó ranking. Verifica que existan configs y modelos para los símbolos solicitados.")
        return
    df = pd.DataFrame(records).sort_values("delta_1d", ascending=False).reset_index(drop=True)
    print("Ranking Equity\n", df.to_string(index=False))
    summary = summarize(df)
    if summary:
        print("\n" + summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        if summary:
            with open(args.output + ".txt", "w", encoding="utf-8") as fh:
                fh.write(summary + "\n")
        logging.info("Ranking guardado en %s", args.output)


if __name__ == "__main__":
    main()
