#!/usr/bin/env python3
"""Orquesta fetch -> merge -> train para todos los símbolos configurados."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assets import DEFAULT_COINGECKO_IDS, DEFAULT_CRYPTO_SYMBOLS  # noqa: E402

DATASETS_DIR = Path("data/datasets")
MARKET_DIR = Path("data/market")
EQUITY_MARKET_DIR = Path("data/market_equities")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline completo de ingestión y entrenamiento.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_CRYPTO_SYMBOLS,
        help="Símbolos Binance (USDT).",
    )
    parser.add_argument(
        "--coin-ids",
        nargs="*",
        default=DEFAULT_COINGECKO_IDS,
        help="IDs de CoinGecko alineados con los símbolos.",
    )
    parser.add_argument("--interval", default=None, help="Intervalo único (obsoleto, usa --intervals).")
    parser.add_argument(
        "--intervals",
        nargs="*",
        default=None,
        help="Lista de intervalos para procesar (ej. 5m 15m 1h).",
    )
    parser.add_argument("--days", type=int, default=365, help="Días hacia atrás al ejecutar fetch.")
    parser.add_argument("--depth-limit", type=int, default=100, help="Niveles del order book.")
    parser.add_argument(
        "--rss-feeds",
        nargs="*",
        default=["https://www.coindesk.com/arc/outboundfeeds/rss/"],
        help="Feeds RSS para scrapeo básico.",
    )
    parser.add_argument(
        "--onchain-metrics",
        nargs="*",
        default=["transactions-per-second", "market-price"],
        help="Series on-chain a descargar en blockchain.com.",
    )
    parser.add_argument("--onchain-timespan", default="60days", help="Timespan para métricas on-chain.")
    parser.add_argument(
        "--asset-family",
        choices=["crypto", "equity"],
        default="crypto",
        help="Define la familia de activos para el pipeline completo.",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost", "lightgbm"],
        default="random_forest",
        help="Tipo de modelo para la fase de entrenamiento.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="^GSPC",
        help="Benchmark para enriquecer features de equity.",
    )
    parser.add_argument(
        "--benchmark-days",
        type=int,
        default=365,
        help="Histórico en días para el benchmark (equity).",
    )
    parser.add_argument(
        "--macro-symbols",
        nargs="*",
        default=["^GSPC", "^IXIC"],
        help="Símbolos macro/sectoriales adicionales para equity.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Omitir la fase de descarga.")
    parser.add_argument("--skip-merge", action="store_true", help="Omitir la fase de merge.")
    parser.add_argument("--skip-train", action="store_true", help="Omitir entrenamiento.")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging del orquestador.")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    logging.info("Ejecutando: %s", " ".join(cmd))
    subprocess.run([sys.executable] + cmd, check=True)


def latest_dataset(symbol: str, interval: str, asset_family: str) -> Path:
    pattern = DATASETS_DIR.glob(f"{symbol}_{asset_family}_{interval}_*rows.parquet")
    files = list(pattern)
    if not files:
        raise FileNotFoundError(f"Dataset no encontrado para {symbol} {interval} en {DATASETS_DIR}")
    return max(files, key=os.path.getmtime)


def market_exists(symbol: str, interval: str, asset_family: str) -> bool:
    base_dir = MARKET_DIR if asset_family == "crypto" else EQUITY_MARKET_DIR
    prefix = "binance" if asset_family == "crypto" else "equity"
    pattern = base_dir.glob(f"{prefix}_{symbol}_{interval}_*.parquet")
    return any(pattern)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    intervals = args.intervals or ([args.interval] if args.interval else ["15m"])

    if not args.skip_fetch:
        cmd = [
            "scripts/fetch_public_data.py",
            "--days",
            str(args.days),
            "--intervals",
            *intervals,
            "--symbols",
            *args.symbols,
            "--coin-ids",
            *args.coin_ids,
            "--include-order-book",
            "--depth-limit",
            str(args.depth_limit),
            "--rss-feeds",
            *args.rss_feeds,
            "--onchain-metrics",
            *args.onchain_metrics,
            "--onchain-timespan",
            args.onchain_timespan,
        ]
        cmd += ["--asset-family", args.asset_family]
        run_cmd(cmd)
    else:
        logging.info("Saltando fase fetch.")

    if not args.skip_merge:
        for interval in intervals:
            symbols_with_data = [sym for sym in args.symbols if market_exists(sym, interval, args.asset_family)]
            missing = sorted(set(args.symbols) - set(symbols_with_data))
            if missing:
                logging.warning(
                    "Se omiten merge/train para %s en intervalo %s (sin velas en data/market).",
                    ", ".join(missing),
                    interval,
                )
            for symbol in symbols_with_data:
                cmd = [
                    "scripts/merge_history.py",
                    "--symbol",
                    symbol,
                    "--interval",
                    interval,
                    "--include-order-book",
                    "--include-onchain",
                    "--include-spot",
                    "--asset-family",
                    args.asset_family,
                ]
                if args.asset_family == "equity":
                    cmd += [
                        "--benchmark-symbol",
                        args.benchmark_symbol,
                        "--benchmark-days",
                        str(args.benchmark_days),
                        "--macro-symbols",
                        *args.macro_symbols,
                    ]
                run_cmd(cmd)
    else:
        logging.info("Saltando fase merge.")

    if not args.skip_train:
        for interval in intervals:
            for symbol in args.symbols:
                try:
                    dataset = latest_dataset(symbol, interval, args.asset_family)
                except FileNotFoundError:
                    logging.warning(
                        "Sin dataset para %s (%s). ¿Se saltó merge por falta de datos?",
                        symbol,
                        interval,
                    )
                    continue
                cmd = [
                    "scripts/train_price_model.py",
                    "--dataset",
                    str(dataset),
                    "--symbol",
                    symbol,
                    "--interval",
                    interval,
                    "--horizon",
                    "28",
                    "--asset-family",
                    args.asset_family,
                    "--model-type",
                    args.model_type,
                ]
                run_cmd(cmd)
    else:
        logging.info("Saltando fase train.")

    logging.info("Pipeline finalizado.")


if __name__ == "__main__":
    main()
