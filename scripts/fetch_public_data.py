#!/usr/bin/env python3
"""Descarga datos públicos de Binance y CoinGecko y los almacena localmente."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_sources.binance import fetch_klines, fetch_klines_range, fetch_order_book  # noqa: E402
from app.data_sources.coingecko import fetch_market_overview, fetch_status_updates  # noqa: E402
from app.data_sources.blockchain_com import fetch_chart  # noqa: E402
from app.data_sources.rss import fetch_feed  # noqa: E402
from app.data_sources.twitter_api import fetch_recent_tweets  # noqa: E402
from app.data_sources.reddit_api import fetch_subreddit_posts  # noqa: E402
from app.data_sources.coinmarketcal import fetch_events as fetch_coinmarketcal_events  # noqa: E402
from app.data_sources.glassnode import fetch_metric as fetch_glassnode_metric  # noqa: E402
from app.data_sources.coinmetrics import fetch_asset_metrics  # noqa: E402
from app.data_sources.yfinance_equities import fetch_equity_history  # noqa: E402
from app.assets import DEFAULT_COINGECKO_IDS, DEFAULT_CRYPTO_SYMBOLS  # noqa: E402

DATA_DIR = Path("data")
MARKET_DIR = DATA_DIR / "market"
EQUITY_MARKET_DIR = DATA_DIR / "market_equities"
SPOT_DIR = DATA_DIR / "spot"
TEXT_DIR = DATA_DIR / "text"
LIQUIDITY_DIR = DATA_DIR / "liquidity"
ONCHAIN_DIR = DATA_DIR / "onchain"
NEWS_DIR = DATA_DIR / "news"
SENTIMENT_DIR = DATA_DIR / "sentiment"
EVENTS_DIR = DATA_DIR / "events"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingesta pública (Binance + CoinGecko) para alimentar historiales locales."
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_CRYPTO_SYMBOLS,
        help="Símbolos Binance (formato USDT).",
    )
    parser.add_argument(
        "--asset-family",
        choices=["crypto", "equity"],
        default="crypto",
        help="Familia de activos a descargar.",
    )
    parser.add_argument(
        "--interval",
        default=None,
        help="Intervalo único (obsoleto, usa --intervals).",
    )
    parser.add_argument(
        "--intervals",
        nargs="*",
        default=None,
        help="Lista de intervalos (1m,5m,15m,1h,...).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Días hacia atrás para reconstruir velas (paginará automáticamente).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Número de velas a descargar (se usa sólo si --days es 0).",
    )
    parser.add_argument(
        "--coin-ids",
        nargs="*",
        default=DEFAULT_COINGECKO_IDS,
        help="IDs de CoinGecko para el overview.",
    )
    parser.add_argument(
        "--vs-currency",
        default="usd",
        help="Moneda de referencia para CoinGecko.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging.",
    )
    parser.add_argument(
        "--include-order-book",
        action="store_true",
        help="Descargar también snapshot del order book (depth).",
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=100,
        help="Número de niveles del libro de órdenes.",
    )
    parser.add_argument(
        "--rss-feeds",
        nargs="*",
        default=["https://www.coindesk.com/arc/outboundfeeds/rss/"],
        help="Feeds RSS públicos para headlines.",
    )
    parser.add_argument(
        "--onchain-metrics",
        nargs="*",
        default=["transactions-per-second", "market-price", "hash-rate"],
        help="Series on-chain disponibles en blockchain.com/charts.",
    )
    parser.add_argument(
        "--onchain-timespan",
        default="30days",
        help="Timespan para las métricas on-chain (ej. 30days, 60days).",
    )
    parser.add_argument(
        "--enable-twitter",
        action="store_true",
        help="Descargar tweets recientes usando TWITTER_BEARER_TOKEN.",
    )
    parser.add_argument(
        "--twitter-query",
        default="(bitcoin OR crypto) lang:en -is:retweet",
        help="Query de Twitter (sintaxis API v2).",
    )
    parser.add_argument(
        "--twitter-max",
        type=int,
        default=50,
        help="Máximo de tweets a recuperar.",
    )
    parser.add_argument(
        "--enable-reddit",
        action="store_true",
        help="Descargar posts recientes de subreddits públicos.",
    )
    parser.add_argument(
        "--reddit-subreddits",
        nargs="*",
        default=["CryptoCurrency", "Bitcoin", "ethtrader"],
        help="Lista de subreddits a consultar.",
    )
    parser.add_argument(
        "--reddit-limit",
        type=int,
        default=25,
        help="Posts por subreddit.",
    )
    parser.add_argument(
        "--enable-coinmarketcal",
        action="store_true",
        help="Recuperar eventos de CoinMarketCal (requiere API key/secret).",
    )
    parser.add_argument(
        "--coinmarketcal-max",
        type=int,
        default=20,
        help="Número máximo de eventos CoinMarketCal.",
    )
    parser.add_argument(
        "--glassnode-metrics",
        nargs="*",
        default=["addresses/active_count"],
        help="Métricas avanzadas de Glassnode (requiere GLASSNODE_API_KEY).",
    )
    parser.add_argument(
        "--glassnode-assets",
        nargs="*",
        default=["BTC"],
        help="Activos (tickers) para métricas Glassnode.",
    )
    parser.add_argument(
        "--glassnode-days",
        type=int,
        default=90,
        help="Ventana en días para Glassnode.",
    )
    parser.add_argument(
        "--coinmetrics-assets",
        nargs="*",
        default=["btc"],
        help="Activos CoinMetrics (tickers en minúscula).",
    )
    parser.add_argument(
        "--coinmetrics-metrics",
        nargs="*",
        default=["PriceUSD", "TxCnt"],
        help="Métricas CoinMetrics (requiere COINMETRICS_API_KEY).",
    )
    parser.add_argument(
        "--coinmetrics-days",
        type=int,
        default=90,
        help="Ventana temporal para CoinMetrics.",
    )
    parser.add_argument(
        "--coinmetrics-frequency",
        default="1d",
        help="Frecuencia (1d, 1h, etc.) para CoinMetrics.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for directory in (
        MARKET_DIR,
        EQUITY_MARKET_DIR,
        SPOT_DIR,
        TEXT_DIR,
        LIQUIDITY_DIR,
        ONCHAIN_DIR,
        NEWS_DIR,
        SENTIMENT_DIR,
        EVENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """Convierte una cadena en un slug seguro para usar en nombres de archivo."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def store_dataframe(frame: pd.DataFrame, base_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"{prefix}_{timestamp}.parquet"
    frame.to_parquet(path, index=False)
    return path


def store_market_by_month(frame: pd.DataFrame, symbol: str, interval: str, base_dir: Path, prefix: str) -> List[Path]:
    if frame.empty:
        return []
    frame = frame.sort_values("open_time").copy()
    frame["month"] = frame["open_time"].dt.to_period("M")
    written: List[Path] = []
    for period, chunk in frame.groupby("month"):
        fname = f"{prefix}_{symbol}_{interval}_{period.strftime('%Y%m')}.parquet"
        path = base_dir / fname
        chunk.drop(columns=["month"]).to_parquet(path, index=False)
        written.append(path)
    return written


def store_order_book(data: Dict[str, Any], base_dir: Path, prefix: str) -> Path:
    bids = pd.DataFrame(data.get("bids", []), columns=["price", "quantity"])
    bids["side"] = "bid"
    asks = pd.DataFrame(data.get("asks", []), columns=["price", "quantity"])
    asks["side"] = "ask"
    frame = pd.concat([bids, asks], ignore_index=True)
    frame["price"] = frame["price"].astype(float)
    frame["quantity"] = frame["quantity"].astype(float)
    frame["symbol"] = data["symbol"]
    frame["last_update_id"] = data["last_update_id"]
    frame["snapshot_ts"] = data["timestamp"]
    return store_dataframe(frame, base_dir, prefix)


def run_crypto_ingestion(args: argparse.Namespace) -> None:
    selected_intervals = args.intervals or ([args.interval] if args.interval else ["15m"])

    for interval in selected_intervals:
        for symbol in args.symbols:
            logging.info("Descargando velas de Binance para %s (%s)", symbol, interval)
            try:
                if args.days and args.days > 0:
                    candles = fetch_klines_range(symbol, interval=interval, days=args.days)
                else:
                    candles = fetch_klines(symbol, interval=interval, limit=args.limit or 500)
            except requests.HTTPError as exc:
                logging.error(
                    "Falló la descarga de %s (%s): %s. Verifica que el símbolo exista en Binance.",
                    symbol,
                    interval,
                    exc,
                )
                continue
            written = store_market_by_month(candles, symbol, interval, MARKET_DIR, "binance")
            logging.info(
                "Guardadas %s filas en %s archivos mensuales (intervalo %s)", len(candles), len(written), interval
            )
            if args.include_order_book:
                depth = fetch_order_book(symbol, limit=args.depth_limit)
                ob_file = store_order_book(depth, LIQUIDITY_DIR, f"orderbook_{symbol}_{interval}")
                logging.info("Snapshot de order book guardado en %s", ob_file)

    logging.info("Descargando overview CoinGecko para %s", ", ".join(args.coin_ids))
    markets = fetch_market_overview(args.coin_ids, vs_currency=args.vs_currency)
    market_file = store_dataframe(markets, SPOT_DIR, "coingecko_markets")
    logging.info("Guardado overview en %s", market_file)

    updates = fetch_status_updates()
    if not updates.empty:
        text_file = store_dataframe(updates, TEXT_DIR, "coingecko_status_updates")
        logging.info("Guardadas %s noticias en %s", len(updates), text_file)
    else:
        logging.info("CoinGecko no devolvió actualizaciones en este momento.")

    for feed in args.rss_feeds:
        news = fetch_feed(feed)
        if news.empty:
            logging.info("Feed RSS sin entradas (%s)", feed)
            continue
        slug = slugify(feed)
        news_file = store_dataframe(news, NEWS_DIR, f"rss_{slug}")
        logging.info("Guardadas %s entradas del feed %s en %s", len(news), feed, news_file)

    if args.enable_twitter:
        tweets = fetch_recent_tweets(args.twitter_query, max_results=args.twitter_max)
        if tweets.empty:
            logging.info("Twitter no devolvió resultados (verifica TWITTER_BEARER_TOKEN).")
        else:
            slug = slugify(args.twitter_query)
            twitter_file = store_dataframe(tweets, SENTIMENT_DIR, f"twitter_{slug}")
            logging.info("Guardados %s tweets recientes en %s", len(tweets), twitter_file)

    if args.enable_reddit:
        for subreddit in args.reddit_subreddits:
            posts = fetch_subreddit_posts(subreddit, limit=args.reddit_limit)
            if posts.empty:
                logging.info("Subreddit %s sin posts (o rate limited).", subreddit)
                continue
            reddit_file = store_dataframe(posts, SENTIMENT_DIR, f"reddit_{slugify(subreddit)}")
            logging.info("Guardados %s posts de r/%s en %s", len(posts), subreddit, reddit_file)

    if args.enable_coinmarketcal:
        events = fetch_coinmarketcal_events(max_results=args.coinmarketcal_max)
        if events.empty:
            logging.info("CoinMarketCal no devolvió eventos (¿faltan credenciales?).")
        else:
            events_file = store_dataframe(events, EVENTS_DIR, "coinmarketcal_events")
            logging.info("Guardados %s eventos de CoinMarketCal en %s", len(events), events_file)

    for metric in args.onchain_metrics:
        try:
            chart = fetch_chart(metric, timespan=args.onchain_timespan)
        except requests.HTTPError as exc:  # type: ignore[name-defined]
            logging.warning("No se pudo descargar métrica %s: %s", metric, exc)
            continue
        if chart.empty:
            logging.info("Métrica %s sin datos en este rango.", metric)
            continue
        onchain_file = store_dataframe(chart, ONCHAIN_DIR, f"onchain_{metric}")
        logging.info("Serie on-chain %s guardada en %s", metric, onchain_file)

    for metric in args.glassnode_metrics:
        for asset in args.glassnode_assets:
            data = fetch_glassnode_metric(metric, asset=asset, since_days=args.glassnode_days)
            if data.empty:
                logging.info("Glassnode sin datos para %s (%s) o falta GLASSNODE_API_KEY.", metric, asset)
                continue
            glass_file = store_dataframe(data, ONCHAIN_DIR, f"glassnode_{metric.replace('/', '_')}_{asset}")
            logging.info("Serie Glassnode %s %s guardada en %s", metric, asset, glass_file)

    if args.coinmetrics_metrics and args.coinmetrics_assets:
        cm_frame = fetch_asset_metrics(
            assets=args.coinmetrics_assets,
            metrics=args.coinmetrics_metrics,
            since_days=args.coinmetrics_days,
            frequency=args.coinmetrics_frequency,
        )
        if cm_frame.empty:
            logging.info("CoinMetrics sin datos o falta COINMETRICS_API_KEY.")
        else:
            cm_file = store_dataframe(cm_frame, ONCHAIN_DIR, "coinmetrics_asset_metrics")
            logging.info(
                "Serie CoinMetrics guardada en %s (assets=%s, metrics=%s)",
                cm_file,
                ",".join(args.coinmetrics_assets),
                ",".join(args.coinmetrics_metrics),
            )


def run_equity_ingestion(args: argparse.Namespace) -> None:
    selected_intervals = args.intervals or ([args.interval] if args.interval else ["1d"])
    for interval in selected_intervals:
        for symbol in args.symbols:
            logging.info("Descargando velas de Yahoo Finance para %s (%s)", symbol, interval)
            candles = fetch_equity_history(symbol, interval=interval, days=args.days or 365)
            if candles.empty:
                logging.warning("Sin datos para %s (%s).", symbol, interval)
                continue
            written = store_market_by_month(candles, symbol, interval, EQUITY_MARKET_DIR, "equity")
            logging.info(
                "Guardadas %s filas en %s archivos mensuales (intervalo %s)",
                len(candles),
                len(written),
                interval,
            )


def run_ingestion(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.asset_family == "equity":
        run_equity_ingestion(args)
        return
    run_crypto_ingestion(args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_ingestion(args)


if __name__ == "__main__":
    main()
