#!/usr/bin/env python3
"""Utilidad para gestionar activos cripto (listar y añadir)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import sys
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_FILE = REPO_ROOT / "assets" / "crypto_assets.yaml"
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
LIQUIDITY_DIR = DATA_DIR / "liquidity"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUTS_DIR = REPO_ROOT / "outputs"
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def load_assets() -> List[Dict[str, Any]]:
    if not ASSETS_FILE.exists():
        return []
    return yaml.load(ASSETS_FILE.read_text(encoding="utf-8")) or []


def save_assets(data: List[Dict[str, Any]]) -> None:
    with ASSETS_FILE.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


def list_symbols() -> None:
    assets = load_assets()
    symbols = [asset["symbol"] for asset in assets if asset.get("enabled", False)]
    print(" ".join(symbols))


def list_coin_ids() -> None:
    assets = load_assets()
    ids = [asset["coin_id"] for asset in assets if asset.get("enabled", False)]
    print(" ".join(ids))


def add_crypto(args: argparse.Namespace) -> None:
    assets = load_assets()
    symbol = args.symbol.upper()
    ticker = args.ticker.upper()
    if any(asset["symbol"] == symbol for asset in assets):
        raise SystemExit(f"El símbolo {symbol} ya existe en assets/crypto_assets.yaml")
    entry = {
        "symbol": symbol,
        "ticker": ticker,
        "display": args.display,
        "coin_id": args.coingecko_id,
        "ranking_symbol": args.ranking_symbol or f"{ticker}-USD",
        "enabled": args.enable_default,
        "ranking_enabled": args.ranking_enabled if args.ranking_enabled is not None else args.enable_default,
        "default_texts": [
            args.text_one or f"{args.display} gana tracción entre los traders.",
            args.text_two or f"Inversores evalúan la situación de {args.display}.",
        ],
    }
    assets.append(entry)
    assets.sort(key=lambda item: item["symbol"])
    save_assets(assets)
    generate_configs(entry, overwrite=args.overwrite_configs)
    print(f"Activo {symbol} añadido. Recuerda volver a ejecutar make docker-build si es necesario.")


CONFIG_TEMPLATE = """app:
  domain: "crypto"
  default_symbol: "{ticker}-USD"
  default_texts:
    - "{text_one}"
    - "{text_two}"

sentiment:
  enabled: true
  provider: "huggingface"
  task: "sentiment-analysis"
  model_id: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  device: -1
  batch_size: 16

price:
  enabled: true
  provider: "local_sklearn"
  symbol: "{ticker}-USD"
  market_symbol: "{symbol}"
  interval: "{interval}"
  asset_family: "crypto"
  model_path: "../models/local_price_{symbol}_crypto_{interval}.joblib"
  source: "local"
  lookback_days: {lookback}
  fast_ma: 7
  slow_ma: 21
  rsi_window: 14
  feature_columns: []
  huggingface: null
"""


def generate_configs(entry: Dict[str, Any], overwrite: bool = False) -> None:
    ticker_lower = entry["ticker"].lower()
    texts = entry["default_texts"]
    for interval, lookback in (("5m", 7), ("15m", 30), ("1h", 30)):
        target = CONFIG_DIR / f"local-model-{ticker_lower}-{interval}.yaml"
        if target.exists() and not overwrite:
            continue
        content = CONFIG_TEMPLATE.format(
            ticker=entry["ticker"],
            symbol=entry["symbol"],
            interval=interval,
            lookback=lookback,
            text_one=texts[0],
            text_two=texts[1],
        )
        target.write_text(content, encoding="utf-8")
        print(f"Config generada: {target.relative_to(REPO_ROOT)}")


def delete_glob(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        try:
            path.unlink()
            print(f"Eliminado: {path.relative_to(REPO_ROOT)}")
        except Exception as exc:  # pragma: no cover
            print(f"No se pudo eliminar {path}: {exc}")


def remove_artifacts(symbol: str, ticker: str) -> None:
    ticker_lower = ticker.lower()
    for interval in ("5m", "15m", "1h"):
        cfg = CONFIG_DIR / f"local-model-{ticker_lower}-{interval}.yaml"
        if cfg.exists():
            cfg.unlink()
            print(f"Eliminado config: {cfg.relative_to(REPO_ROOT)}")
    delete_glob(MARKET_DIR, f"binance_{symbol}_*.parquet")
    delete_glob(LIQUIDITY_DIR, f"orderbook_{symbol}_*.parquet")
    delete_glob(DATASETS_DIR, f"{symbol}_*_rows.parquet")
    delete_glob(MODELS_DIR, f"local_price_{symbol}_crypto_*.joblib")
    delete_glob(REPORTS_DIR, f"training_report_{symbol}_crypto_*.json")
    delete_glob(OUTPUTS_DIR, f"*{ticker}*")


def remove_crypto(args: argparse.Namespace) -> None:
    assets = load_assets()
    symbol = args.symbol.upper()
    entry = next((asset for asset in assets if asset["symbol"] == symbol), None)
    if not entry:
        print(f"No se encontró el símbolo {symbol} en assets/crypto_assets.yaml")
        sys.exit(0)
    assets = [asset for asset in assets if asset["symbol"] != symbol]
    save_assets(assets)
    print(f"Activo {symbol} eliminado del inventario.")
    remove_artifacts(symbol, entry["ticker"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gestor de activos cripto.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-symbols")
    sub.add_parser("list-coingecko")

    add = sub.add_parser("add-crypto")
    add.add_argument("--symbol", required=True, help="Símbolo Binance (ej. UNIUSDT).")
    add.add_argument("--ticker", required=True, help="Ticker corto (UNI).")
    add.add_argument("--display", required=True, help="Nombre descriptivo.")
    add.add_argument("--coingecko-id", required=True, help="Coin ID usado por CoinGecko.")
    add.add_argument("--ranking-symbol", default=None, help="Ticker para ranking (ej. UNI-USD).")
    add.add_argument("--enable-default", action="store_true", help="Incluye el símbolo en la lista por defecto.")
    add.add_argument(
        "--ranking-enabled",
        choices=["true", "false"],
        default=None,
        help="Habilita el ranking multi-símbolo (por defecto usa el valor de --enable-default).",
    )
    add.add_argument("--text-one", default=None, help="Texto por defecto 1 para configs.")
    add.add_argument("--text-two", default=None, help="Texto por defecto 2 para configs.")
    add.add_argument("--overwrite-configs", action="store_true", help="Sobrescribe configs existentes.")
    rem = sub.add_parser("remove-crypto")
    rem.add_argument("--symbol", required=True, help="Símbolo Binance (ej. KSMUSDT).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list-symbols":
        list_symbols()
    elif args.command == "list-coingecko":
        list_coin_ids()
    elif args.command == "add-crypto":
        if args.ranking_enabled is not None:
            args.ranking_enabled = args.ranking_enabled.lower() == "true"
        add_crypto(args)
    elif args.command == "remove-crypto":
        remove_crypto(args)


if __name__ == "__main__":
    main()
