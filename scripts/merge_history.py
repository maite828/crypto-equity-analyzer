#!/usr/bin/env python3
"""Une los archivos Parquet generados en data/ en un dataset único listo para entrenamiento."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_sources.yfinance_equities import (
    attach_benchmark_features,
    compute_obv,
    compute_vwap,
    fetch_equity_history,
    fetch_corporate_features,
    fetch_macro_series,
)  # noqa: E402
from app.assets import COINGECKO_MAP  # noqa: E402
DATA_DIR = Path("data")
MARKET_DIR = DATA_DIR / "market"
EQUITY_MARKET_DIR = DATA_DIR / "market_equities"
SPOT_DIR = DATA_DIR / "spot"
LIQUIDITY_DIR = DATA_DIR / "liquidity"
ONCHAIN_DIR = DATA_DIR / "onchain"
NEWS_DIR = DATA_DIR / "news"

OUTPUT_DIR = DATA_DIR / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusiona velas, métricas spot, order book y on-chain en un dataset."
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Símbolo (BTCUSDT para cripto, TSLA para equity).",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        help="Intervalo de mercado que deseas consolidar (debe coincidir con los archivos existentes).",
    )
    parser.add_argument(
        "--asset-family",
        choices=["crypto", "equity"],
        default="crypto",
        help="Fuente base para determinar directorios y features.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="^GSPC",
        help="Benchmark para features de equity.",
    )
    parser.add_argument(
        "--benchmark-days",
        type=int,
        default=365,
        help="Días de histórico para el benchmark (equity).",
    )
    parser.add_argument(
        "--macro-symbols",
        nargs="*",
        default=["^GSPC", "^IXIC"],
        help="Símbolos macro/sectoriales adicionales para equity.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta personalizada para guardar el dataset. Por defecto data/datasets/{symbol}_{interval}.parquet",
    )
    parser.add_argument(
        "--include-order-book",
        action="store_true",
        help="Agrega features de order book (bid/ask imbalance).",
    )
    parser.add_argument(
        "--include-onchain",
        action="store_true",
        help="Une series on-chain (market-price, TPS, etc.) usando timestamp cercano.",
    )
    parser.add_argument(
        "--include-spot",
        action="store_true",
        help="Incluye snapshots de CoinGecko (market cap, volumen).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging.",
    )
    return parser.parse_args()


def load_market(
    symbol: str,
    interval: str,
    asset_family: str,
    benchmark_symbol: Optional[str] = None,
    benchmark_days: int = 365,
    macro_symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    prefix = "binance" if asset_family == "crypto" else "equity"
    base_dir = MARKET_DIR if asset_family == "crypto" else EQUITY_MARKET_DIR
    files = sorted(base_dir.glob(f"{prefix}_{symbol}_{interval}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron velas en {base_dir} para {symbol} {interval}")
    frames = [pd.read_parquet(path) for path in files]
    market = pd.concat(frames, ignore_index=True)
    market = market.drop_duplicates(subset=["open_time"])
    market = market.sort_values("open_time").reset_index(drop=True)
    market["mid_price"] = (market["open"] + market["close"]) / 2
    market["return_1"] = market["close"].pct_change().fillna(0)
    market["return_4"] = market["close"].pct_change(4).fillna(0)
    market["return_16"] = market["close"].pct_change(16).fillna(0)
    market["volatility_16"] = market["return_1"].rolling(window=16).std().fillna(0)
    market["volatility_64"] = market["return_1"].rolling(window=64).std().fillna(0)
    market["ema_fast"] = market["close"].ewm(span=8, adjust=False).mean()
    market["ema_slow"] = market["close"].ewm(span=21, adjust=False).mean()
    market["ema_ratio"] = market["ema_fast"] / (market["ema_slow"] + 1e-9)
    market["rsi_14"] = compute_rsi(market["close"], window=14)
    market["atr_14"] = compute_atr(market, window=14)
    market["close_time"] = market["close_time"].dt.round("1s")
    if asset_family == "equity":
        market = enhance_equity_features(market, interval, benchmark_symbol or "^GSPC", benchmark_days)
        market = attach_macro_series(market, macro_symbols or [], interval, benchmark_days)
        corp = fetch_corporate_features(symbol)
        for key, value in corp.items():
            market[key] = value
    return market


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift()).abs()
    low_close_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def enhance_equity_features(
    df: pd.DataFrame, interval: str, benchmark_symbol: str, benchmark_days: int
) -> pd.DataFrame:
    volume = df.get("volume")
    if volume is not None:
        df["obv"] = compute_obv(df["close"], volume)
    else:
        df["obv"] = 0.0
    df["vwap_20"] = compute_vwap(df, 20)
    df["vwap_ratio"] = df["close"] / (df["vwap_20"] + 1e-9)
    df = attach_benchmark_features(df, benchmark_symbol, interval, benchmark_days)
    df["benchmark_close"] = df.get("benchmark_close", 0.0).fillna(0.0)
    df["benchmark_return_1"] = df.get("benchmark_return_1", 0.0).fillna(0.0)
    df["close_to_benchmark"] = df.get("close_to_benchmark", 1.0).fillna(1.0)
    return df


def attach_macro_series(
    df: pd.DataFrame, symbols: list[str], interval: str, lookback_days: int
) -> pd.DataFrame:
    for symbol in symbols or []:
        try:
            series = fetch_macro_series(symbol, interval, lookback_days)
        except Exception as exc:  # pragma: no cover
            logging.warning("No se pudo descargar macro %s: %s", symbol, exc)
            continue
        df = pd.merge_asof(
            df.sort_values("close_time"),
            series.sort_values("close_time"),
            on="close_time",
            direction="nearest",
        )
        close_col = f"macro_{symbol}_close"
        if close_col in df.columns:
            macro_close = df[close_col].ffill().bfill().fillna(0.0)
            macro_returns = macro_close.pct_change().fillna(0.0)
            df[close_col] = macro_close
            df[f"macro_{symbol}_return_1"] = macro_returns
            df[f"macro_{symbol}_return_5"] = macro_close.pct_change(5).fillna(0.0)
            df[f"macro_{symbol}_volatility_20"] = macro_returns.rolling(20).std().fillna(0.0)
            df[f"macro_{symbol}_corr_20"] = (
                df["close"].rolling(20).corr(macro_close).fillna(0.0)
            )
            df[f"macro_{symbol}_spread"] = df["close"] - macro_close
            df[f"macro_{symbol}_beta_20"] = (
                df["return_1"].rolling(20).cov(macro_returns)
                / (macro_returns.rolling(20).var() + 1e-9)
            ).fillna(0.0)
    return df


def load_order_book(symbol: str) -> pd.DataFrame:
    files = sorted(LIQUIDITY_DIR.glob(f"orderbook_{symbol}_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    depth = pd.concat(frames, ignore_index=True)
    depth["timestamp"] = pd.to_datetime(depth["snapshot_ts"])
    bids = depth[depth["side"] == "bid"].groupby("timestamp")["quantity"].sum()
    asks = depth[depth["side"] == "ask"].groupby("timestamp")["quantity"].sum()
    ask_min = depth[depth["side"] == "ask"].groupby("timestamp")["price"].min()
    bid_max = depth[depth["side"] == "bid"].groupby("timestamp")["price"].max()
    book = pd.DataFrame(
        {
            "bid_volume": bids,
            "ask_volume": asks,
            "spread": ask_min - bid_max,
        }
    )
    book["ob_imbalance"] = (book["bid_volume"] - book["ask_volume"]) / (
        book["bid_volume"] + book["ask_volume"] + 1e-9
    )
    book["ob_pressure"] = book["ob_imbalance"] * (book["bid_volume"] + book["ask_volume"])
    book = book.reset_index()
    book = book.rename(columns={"timestamp": "close_time"})
    return book


def load_spot_metrics() -> pd.DataFrame:
    files = sorted(SPOT_DIR.glob("coingecko_markets_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    markets = pd.concat(frames, ignore_index=True)
    markets["fetched_at"] = pd.to_datetime(markets["fetched_at"])
    markets = markets.sort_values("fetched_at")
    markets["market_cap_change"] = markets.groupby("id")["market_cap"].diff().fillna(0)
    cols = [
        "id",
        "current_price",
        "market_cap",
        "market_cap_change",
        "total_volume",
        "price_change_percentage_24h",
        "price_change_percentage_7d_in_currency",
        "fetched_at",
    ]
    return markets[cols]


def load_onchain(chart: str) -> pd.DataFrame:
    files = sorted(ONCHAIN_DIR.glob(f"onchain_{chart}_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    series = pd.concat(frames, ignore_index=True)
    series["timestamp"] = pd.to_datetime(series["timestamp"])
    return series.rename(columns={"value": f"onchain_{chart}"})


def merge_all(args: argparse.Namespace) -> pd.DataFrame:
    market = load_market(
        args.symbol,
        args.interval,
        args.asset_family,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_days=args.benchmark_days,
        macro_symbols=args.macro_symbols,
    )

    if args.asset_family == "crypto" and args.include_order_book:
        book = load_order_book(args.symbol)
        if not book.empty:
            market = market.merge(book, on="close_time", how="left")

    if args.asset_family == "crypto" and args.include_spot:
        spot = load_spot_metrics()
        if not spot.empty:
            # Map symbol to CoinGecko id heuristically
            coin_id = COINGECKO_MAP.get(args.symbol)
            if coin_id:
                subset = spot[spot["id"] == coin_id]
                if not subset.empty:
                    subset = subset.rename(columns={"fetched_at": "close_time"})
                    market = market.merge(
                        subset[
                            [
                                "close_time",
                                "current_price",
                                "market_cap",
                                "total_volume",
                                "price_change_percentage_24h",
                                "price_change_percentage_7d_in_currency",
                            ]
                        ],
                        on="close_time",
                        how="left",
                    )

    if args.asset_family == "crypto" and args.include_onchain:
        for metric in ["transactions-per-second", "market-price"]:
            series = load_onchain(metric)
            if series.empty:
                continue
            series = series.rename(columns={"timestamp": "close_time"})
            market = market.merge(series[["close_time", f"onchain_{metric}"]], on="close_time", how="left")

    market = market.sort_values("close_time").reset_index(drop=True)
    return market


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    dataset = merge_all(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = (
        Path(args.output)
        if args.output
        else OUTPUT_DIR / f"{args.symbol}_{args.asset_family}_{args.interval}_{len(dataset)}rows.parquet"
    )
    dataset.to_parquet(target_path, index=False)
    logging.info("Dataset guardado en %s (filas=%s)", target_path, len(dataset))


if __name__ == "__main__":
    main()
