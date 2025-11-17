#!/usr/bin/env python3
"""Hyperparameter tuning for equity models using RandomizedSearchCV."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from app.config import load_config

CONFIGS_DIR = Path("configs")
DATASETS_DIR = Path("data/datasets")
PARAMS_DIR = CONFIGS_DIR / "model_params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for equity models.")
    parser.add_argument(
        "--config",
        help="Config YAML (e.g. configs/local-model-tsla-1h.yaml). Si no se pasa, se infiere a partir de símbolo/intervalo.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset parquet (output de merge_history). Si no se pasa, se infiere desde data/datasets.",
    )
    parser.add_argument("--symbol", help="Símbolo (TSLA, BTCUSDT, etc.) para localizar dataset/config.")
    parser.add_argument("--interval", default="1h", help="Intervalo del modelo (1h, 1d, 15m, ...).")
    parser.add_argument(
        "--asset-family",
        choices=["crypto", "equity"],
        default="equity",
        help="Familia del activo (afecta el nombre del dataset/config).",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost", "lightgbm"],
        default="random_forest",
    )
    parser.add_argument("--n-iter", type=int, default=20, help="Number of sampling iterations")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
        help="Horizonte futuro en pasos para construir la variable objetivo (por defecto 28).",
    )
    return parser.parse_args()


def load_dataset(path: str, horizon: int):
    import pandas as pd

    df = pd.read_parquet(path)
    df["target_price"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["target_price"])
    feature_cols = [
        col
        for col in df.columns
        if col not in {"target_price", "target_return"}
        and df[col].dtype in (np.float64, np.float32, np.int64, np.int32)
    ]
    return df[feature_cols].values, df["target_price"].values, feature_cols


def infer_config_path(symbol: str, interval: str, asset_family: str) -> Path:
    if not symbol:
        raise ValueError("Debes indicar --symbol para inferir la configuración.")
    base = symbol.lower()
    if asset_family == "crypto" and base.endswith("usdt"):
        base = base[:-4]
    return CONFIGS_DIR / f"local-model-{base}-{interval}.yaml"


def latest_dataset_path(symbol: str, interval: str, asset_family: str) -> Path:
    if not symbol:
        raise ValueError("Debes indicar --symbol para localizar el dataset.")
    pattern = DATASETS_DIR.glob(f"{symbol}_{asset_family}_{interval}_*rows.parquet")
    candidates = sorted(pattern, key=lambda p: p.stat().st_mtime)  # oldest->newest
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró dataset para {symbol} ({interval}, {asset_family}). Ejecuta merge_history/pipeline primero."
        )
    return candidates[-1]


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    symbol = args.symbol
    interval = args.interval
    asset_family = args.asset_family
    config_path = Path(args.config) if args.config else infer_config_path(symbol, interval, asset_family)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración {config_path}.")
    dataset_path = Path(args.dataset) if args.dataset else latest_dataset_path(symbol, interval, asset_family)
    LOGGER.info("Usando dataset %s", dataset_path)
    LOGGER.info("Configuración base: %s", config_path)
    settings = load_config(config_path)
    LOGGER.info("Modelo configurado para %s (%s)", settings.price.symbol, settings.price.interval)
    horizon = args.horizon or 28
    X, y, feature_cols = load_dataset(str(dataset_path), horizon)
    if args.model_type == "xgboost":
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=args.random_state)
        param_distributions = {
            "n_estimators": np.arange(200, 600),
            "max_depth": np.arange(3, 12),
            "learning_rate": np.linspace(0.01, 0.15, 15),
            "subsample": np.linspace(0.5, 1.0, 6),
            "colsample_bytree": np.linspace(0.5, 1.0, 6),
        }
    elif args.model_type == "lightgbm":
        model = LGBMRegressor(random_state=args.random_state)
        param_distributions = {
            "n_estimators": np.arange(200, 600),
            "max_depth": np.arange(3, 12),
            "learning_rate": np.linspace(0.01, 0.15, 15),
            "subsample": np.linspace(0.5, 1.0, 6),
            "colsample_bytree": np.linspace(0.5, 1.0, 6),
        }
    else:
        model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
        param_distributions = {
            "n_estimators": np.arange(200, 600),
            "max_depth": np.arange(5, 20),
            "min_samples_leaf": np.arange(1, 5),
        }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        cv=3,
        random_state=args.random_state,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X, y)
    output = {
        "model_type": args.model_type,
        "best_params": _to_serializable(search.best_params_),
        "best_score": float(search.best_score_),
    }
    config_name = config_path.stem
    out_path = PARAMS_DIR / f"{config_name}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Best params saved to {out_path}")


if __name__ == "__main__":
    main()
