#!/usr/bin/env python3
"""Analiza múltiples símbolos de criptomonedas en una sola ejecución."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings, load_config, override_symbol  # noqa: E402
from app.pricing import PriceService  # noqa: E402
from app.sentiment import SentimentService  # noqa: E402
from app.assets import CRYPTO_RANKING_SYMBOLS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 1 (sentimiento) + Paso 2 (ranking de precios) para una lista de símbolos."
    )
    parser.add_argument(
        "--config",
        default="configs/sample-config.yaml",
        help="Archivo de configuración base.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=CRYPTO_RANKING_SYMBOLS,
        help="Lista de símbolos (Yahoo Finance) a analizar.",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="Textos para el análisis de sentimiento (sobre-escribe defaults).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Nivel de logging (INFO, DEBUG...).",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Deshabilita el Paso 1 (sentimiento).",
    )
    parser.add_argument(
        "--skip-price",
        action="store_true",
        help="Deshabilita el Paso 2 (predicciones/ranking).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta opcional a CSV/Parquet para guardar el ranking de precios.",
    )
    return parser.parse_args()


def gather_texts(args: argparse.Namespace, settings: Settings) -> List[str]:
    texts = args.texts or settings.app.default_texts or []
    env_texts = os.getenv("TEXTS")
    if env_texts:
        texts.extend(t.strip() for t in env_texts.split("||") if t.strip())
    seen: set[str] = set()
    ordered: List[str] = []
    for text in texts:
        if text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered


def print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def to_binance_symbol(symbol: str) -> str:
    candidate = symbol.replace("-", "").replace("/", "").upper()
    if candidate.endswith("USD") and not candidate.endswith("USDT"):
        candidate = f"{candidate}T"
    return candidate


def enrich_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if {"price_t", "price_t_plus_1", "price_t_plus_7"}.issubset(df.columns):
        df["expected_price_1d"] = df["price_t_plus_1"]
        df["expected_price_7d"] = df["price_t_plus_7"]
        df["delta_1d"] = df["expected_price_1d"] - df["price_t"]
        df["delta_7d"] = df["expected_price_7d"] - df["price_t"]
    sort_col = "delta_7d" if "delta_7d" in df.columns else None
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    df = df.rename(
        columns={
            "symbol": "asset",
            "price_t": "current_price",
        }
    )
    selected_cols = [
        "asset",
        "current_price",
        "expected_price_1d",
        "expected_price_7d",
        "delta_1d",
        "delta_7d",
    ]
    return df[[col for col in selected_cols if col in df.columns]]


def summarize_rankings(df: pd.DataFrame) -> str | None:
    if "delta_7d" not in df.columns or df.empty:
        return None
    best = df.iloc[0]
    best_delta = best["delta_7d"]
    delta1 = best.get("delta_1d")
    if best_delta > 0:
        lines = [
            f"El modelo prioriza {best['asset']} con una variación esperada de "
            f"{best_delta:+.3f} puntos a 7 días."
        ]
        if isinstance(delta1, (int, float)):
            trend = "reforzando" if delta1 > 0 else "tras un retroceso a 1 día"
            lines.append(
                f"A 1 día el delta es {delta1:+.3f}, {trend} la señal semanal."
            )
        positives = df[df["delta_7d"] > 0]
        if len(positives) > 1:
            runner = positives.iloc[1]
            lines.append(
                f"Como segunda opción aparece {runner['asset']} con "
                f"{runner['delta_7d']:+.3f} a 7 días."
            )
        else:
            lines.append("El resto de símbolos muestran presiones bajistas en este corte.")
    else:
        lines = [
            "Todas las variaciones a 7 días son negativas; el modelo sugiere prudencia "
            "hasta que aparezcan deltas positivos."
        ]
    return " ".join(lines)


def export_frame(df: pd.DataFrame, destination: Optional[str], summary: Optional[str]) -> None:
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        if summary:
            (path.with_suffix(path.suffix + ".txt")).write_text(summary, encoding="utf-8")
    else:
        df.to_csv(path, index=False)
        if summary:
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n# Interpretación automática: ")
                fh.write(summary)
    logging.info("Resultados guardados en %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = load_config(args.config)
    texts = gather_texts(args, settings)

    if settings.sentiment.enabled and not args.skip_sentiment:
        if texts:
            sentiment_service = SentimentService(settings.sentiment)
            sentiment_df = sentiment_service.analyze(texts)
            print_section("PASO 1 · Sentimiento agregado", sentiment_df)
        else:
            logging.info("PASO 1 omitido: no se proporcionaron textos.")

    price_df: Optional[pd.DataFrame] = None
    if settings.price.enabled and not args.skip_price:
        records = []
        for symbol in args.symbols:
            run_settings = override_symbol(settings, symbol)
            try:
                market_symbol = to_binance_symbol(symbol)
                price_cfg = replace(run_settings.price, market_symbol=market_symbol)
                price_service = PriceService(price_cfg)
                series = price_service.predict(run_settings.price.symbol)
                row = {"symbol": symbol}
                row.update(series.to_dict())
                records.append(row)
            except Exception as exc:  # noqa: BLE001
                logging.error("Fallo procesando %s: %s", symbol, exc)
        if records:
            price_df = enrich_price_frame(pd.DataFrame.from_records(records))
            print_section("PASO 2 · Ranking por proyección", price_df)
            summary = summarize_rankings(price_df)
            if summary:
                print(f"\nInterpretación automática:\n{summary}")
            export_frame(price_df, args.output, summary)
        else:
            logging.warning("No se pudieron generar predicciones para los símbolos indicados.")


if __name__ == "__main__":
    main()
