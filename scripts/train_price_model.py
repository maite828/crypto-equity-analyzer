#!/usr/bin/env python3
"""Entrena un modelo local (sklearn) usando los datasets creados por merge_history."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo local de predicción de precios."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta al archivo Parquet consolidado (merge_history).",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Símbolo asociado (para nombrar el modelo).",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        help="Intervalo de las velas usadas.",
    )
    parser.add_argument(
        "--asset-family",
        choices=["crypto", "equity"],
        default="crypto",
        help="Familia del activo (afecta la ruta del modelo).",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Tipo de modelo a entrenar.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
        help="Número de pasos futuro que se quiere predecir (28 × 15m ≈ 7h).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del dataset para evaluación.",
    )
    parser.add_argument(
        "--walk-splits",
        type=int,
        default=5,
        help="Número de splits para validación walk-forward (>=2 habilita el cálculo).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla del RandomForest.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Número de árboles del RandomForest (auto según familia si no se especifica).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidad máxima del RandomForest (auto según familia si no se especifica).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate para XGBoost (por defecto 0.05).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging.",
    )
    return parser.parse_args()


def build_features(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = data.copy()
    df["target_price"] = df["close"].shift(-horizon)
    df["target_return"] = (df["target_price"] - df["close"]) / df["close"]
    df = df.dropna(subset=["target_price"])
    return df


def walk_forward_metrics(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    default_estimators: int,
    default_depth: int,
) -> tuple[Optional[float], Optional[float]]:
    if args.walk_splits < 2:
        return None, None
    tscv = TimeSeriesSplit(n_splits=args.walk_splits)
    maes: List[float] = []
    rmses: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        model = create_model(args, default_estimators, default_depth)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], preds))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
    return (float(np.mean(maes)), float(np.mean(rmses)))


def create_model(args: argparse.Namespace, default_estimators: int, default_depth: int):
    if args.model_type == "xgboost":
        return XGBRegressor(
            n_estimators=args.n_estimators or default_estimators,
            max_depth=args.max_depth or default_depth,
            learning_rate=args.learning_rate or 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=args.random_state,
        )
    return RandomForestRegressor(
        n_estimators=args.n_estimators or default_estimators,
        max_depth=args.max_depth or default_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    df = pd.read_parquet(args.dataset)
    dataset = build_features(df, args.horizon)

    feature_cols = [
        col
        for col in dataset.columns
        if col not in {"target_price", "target_return"}
        and dataset[col].dtype in (np.float64, np.float32, np.int64, np.int32)
    ]
    X = dataset[feature_cols].values
    y = dataset["target_price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        shuffle=False,
    )

    default_estimators = 400 if args.asset_family == "equity" else 300
    default_depth = 10 if args.asset_family == "equity" else 8
    walk_mae, walk_rmse = walk_forward_metrics(X, y, args, default_estimators, default_depth)

    model = create_model(args, default_estimators, default_depth)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"local_price_{args.symbol}_{args.asset_family}_{args.interval}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "horizon": args.horizon,
            "symbol": args.symbol,
            "interval": args.interval,
        },
        model_path,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"training_report_{args.symbol}_{args.asset_family}_{args.interval}.json"
    metrics = {
        "mae": mae,
        "rmse": float(rmse),
        "rows": len(dataset),
        "features": len(feature_cols),
    }
    if walk_mae is not None:
        metrics["walk_mae"] = float(walk_mae)
    if walk_rmse is not None:
        metrics["walk_rmse"] = float(walk_rmse)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logging.info("Modelo guardado en %s", model_path)
    logging.info("MAE=%.4f | RMSE=%.4f | rows=%s", mae, rmse, len(dataset))
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    train_model(args)


if __name__ == "__main__":
    main()
