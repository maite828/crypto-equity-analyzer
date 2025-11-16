"""Módulo encargado de preparar features y ejecutar modelos de precios."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf
from huggingface_hub import hf_hub_download
from joblib import load as joblib_load
from torch import nn

from .config import PriceSection
from .features.local_features import build_feature_row
from .features.equity_features import build_equity_feature_row

LOGGER = logging.getLogger("app.pricing")
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"


class PriceService:
    """Resuelve el proveedor configurado y expone una interfaz única."""

    def __init__(self, config: PriceSection) -> None:
        self.config = config
        if not config.enabled:
            raise ValueError("El módulo de precios está deshabilitado en la configuración.")

        provider = (config.provider or "").lower()
        if provider == "sklearn":
            self.engine = SklearnPriceModel(config)
        elif provider == "local_sklearn":
            self.engine = LocalSklearnPriceModel(config)
        elif provider == "huggingface_gru":
            self.engine = HuggingFaceGRUPriceModel(config)
        else:
            raise NotImplementedError(
                f"Proveedor de precios '{config.provider}' no está soportado."
            )

    def predict(self, symbol: str) -> pd.Series:
        return self.engine.predict(symbol)


class SklearnPriceModel:
    """Genera indicadores técnicos y evalúa un estimador clásico de scikit-learn."""

    def __init__(self, config: PriceSection) -> None:
        self.config = config
        model_path = self._resolve_model_path()
        LOGGER.info("Cargando modelo de precios desde %s", model_path)
        self.model = joblib_load(model_path)

    def _resolve_model_path(self) -> Path:
        if self.config.source == "local":
            if not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo de modelo: {self.config.model_path}\n"
                    "Entrena un estimador de scikit-learn y guarda el artefacto con joblib.dump()."
                )
            return self.config.model_path

        if self.config.source == "huggingface":
            if not self.config.huggingface:
                raise ValueError(
                    "La sección price.source='huggingface' requiere definir el bloque 'huggingface'."
                )
            hf = self.config.huggingface
            token = os.getenv(hf.token_env_var) if hf.token_env_var else None
            LOGGER.info(
                "Descargando modelo desde Hugging Face (%s/%s)",
                hf.repo_id,
                hf.filename,
            )
            downloaded = hf_hub_download(
                repo_id=hf.repo_id,
                filename=hf.filename,
                revision=hf.revision,
                subfolder=hf.subfolder,
                cache_dir=hf.cache_dir,
                token=token,
            )
            return Path(downloaded)

        raise ValueError(f"Fuente de modelo desconocida: {self.config.source}")

    def _download_market_data(self, symbol: str) -> pd.DataFrame:
        LOGGER.info("Descargando %d días de datos para %s", self.config.lookback_days, symbol)
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=self.config.lookback_days)
        data = yf.download(
            tickers=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            raise RuntimeError(f"No se obtuvieron datos desde Yahoo Finance para {symbol}.")
        return data

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data["Close"]
        returns = closes.pct_change()
        feats = pd.DataFrame(index=data.index)
        feats["return_1d"] = returns
        feats["return_7d"] = closes.pct_change(7)
        feats["volatility_7d"] = returns.rolling(window=7).std()
        feats["fast_ma"] = closes.rolling(window=self.config.fast_ma).mean()
        feats["slow_ma"] = closes.rolling(window=self.config.slow_ma).mean()
        feats["ma_ratio"] = feats["fast_ma"] / feats["slow_ma"]
        feats["rsi"] = self._compute_rsi(closes, self.config.rsi_window)
        feats = feats.dropna()
        if feats.empty:
            raise RuntimeError("No se pudieron generar features suficientes (NaN tras rolling).")
        return feats

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def predict(self, symbol: str) -> pd.Series:
        data = self._download_market_data(symbol)
        feats = self._build_features(data)
        latest = feats.iloc[[-1]][self.config.feature_columns]
        LOGGER.info("Generando predicción para %s", symbol)
        prediction = self.model.predict(latest)[0]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(latest)[0].max()
            return pd.Series({"prediction": prediction, "confidence": proba})
        return pd.Series({"prediction": prediction})


class LocalSklearnPriceModel:
    """Evalúa un modelo sklearn entrenado localmente con datasets enriquecidos."""

    def __init__(self, config: PriceSection) -> None:
        self.config = config
        self.asset_family = getattr(config, "asset_family", "crypto")
        if not config.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo local: {config.model_path}. Ejecuta scripts/train_price_model.py primero."
            )
        artifact = joblib_load(config.model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.horizon = artifact.get("horizon", 0)

    def predict(self, symbol: str) -> pd.Series:
        LOGGER.info("Construyendo features locales para %s (%s)", symbol, self.config.interval)
        if self.asset_family == "equity":
            row = build_equity_feature_row(self.config, self.feature_columns)
        else:
            row = build_feature_row(self.config, self.feature_columns)
        prediction = float(self.model.predict(row.values)[0])
        return pd.Series(
            {
                "predicted_price": prediction,
                "horizon_steps": self.horizon,
            }
        )


class SequenceRegressor(nn.Module):
    """Red GRU bidireccional utilizada por el checkpoint público."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        head_names: Iterable[str],
    ) -> None:
        super().__init__()
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        projection_in = hidden_size * (2 if bidirectional else 1)
        self.fc_features = nn.Sequential(
            nn.Linear(projection_in, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in head_names})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Devuelve predicciones por horizonte."""
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        features = self.fc_features(last_step)
        return {name: head(features).squeeze(-1) for name, head in self.heads.items()}


class HuggingFaceGRUPriceModel:
    """Carga el modelo btc-predictor-v4 (minhquan2310) y genera pronósticos multihorizonte."""

    OUTPUT_ALIASES: Mapping[str, str] = {
        "fc_t": "price_t",
        "fc_t_plus_1": "price_t_plus_1",
        "fc_t_plus_7": "price_t_plus_7",
    }

    def __init__(self, config: PriceSection) -> None:
        self.config = config
        hf_cfg = config.huggingface
        if config.provider != "huggingface_gru":
            raise ValueError("Configuración incompatible para HuggingFace GRU.")
        if not hf_cfg:
            raise ValueError("Debes definir price.huggingface para usar el proveedor Hugging Face.")

        self.token = os.getenv(hf_cfg.token_env_var) if hf_cfg.token_env_var else None
        self.repo_id = hf_cfg.repo_id
        self.hf_revision = hf_cfg.revision
        self.hf_subfolder = hf_cfg.subfolder
        self.hf_cache = hf_cfg.cache_dir
        self.model_ckpt = self._download_file(hf_cfg.filename)

        scalers_file = hf_cfg.scalers_file or "preprocessor/scalers.joblib"
        preproc_file = hf_cfg.preproc_file or "preprocessor/preproc_config.joblib"
        self.scalers_path = self._download_file(scalers_file)
        self.preproc_path = self._download_file(preproc_file)

        self.scalers: Dict[str, object] = joblib_load(self.scalers_path)
        self.preproc_config: Dict[str, object] = joblib_load(self.preproc_path)
        self.feature_columns: List[str] = list(
            self.preproc_config.get("feature_columns", ["open", "high", "low", "close"])
        )
        self.sequence_length: int = int(self.preproc_config.get("sequence_length", 7))
        self.log_columns: List[str] = list(
            self.preproc_config.get("log_transform_columns", [])
        )
        self.target_column: str = str(self.preproc_config.get("target_column", "close"))

        self.model = self._load_torch_model()
        self.model.eval()

    def _download_file(self, filename: Optional[str]) -> Path:
        if not filename:
            raise ValueError("El archivo de Hugging Face no puede ser nulo.")
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.hf_revision,
                subfolder=self.hf_subfolder,
                cache_dir=self.hf_cache,
                token=self.token,
            )
        )

    def _load_torch_model(self) -> SequenceRegressor:
        # El checkpoint incluye objetos de OmegaConf; forzamos weights_only=False
        # porque conocemos la procedencia (repositorio público de Hugging Face).
        checkpoint = torch.load(
            self.model_ckpt,
            map_location="cpu",
            weights_only=False,
        )
        model_params = checkpoint.get("hyper_parameters", {}).get(
            "model",
            {"hidden_size": 128, "num_layers": 2, "dropout": 0.0, "bidirectional": True},
        )
        state_dict = checkpoint["state_dict"]
        head_names = [
            name
            for name in self.OUTPUT_ALIASES.keys()
            if f"{name}.weight" in state_dict
        ]
        if not head_names:
            head_names = sorted(
                {
                    key.split(".")[0]
                    for key in state_dict.keys()
                    if key.startswith("fc_") and not key.startswith("fc_features")
                }
            )
        regressor = SequenceRegressor(
            input_size=len(self.feature_columns),
            hidden_size=model_params.get("hidden_size", 128),
            num_layers=model_params.get("num_layers", 2),
            dropout=model_params.get("dropout", 0.0),
            bidirectional=model_params.get("bidirectional", False),
            head_names=head_names,
        )
        regressor.load_state_dict(state_dict, strict=False)
        return regressor

    def predict(self, symbol: str) -> pd.Series:
        series = self._prepare_sequence(symbol)
        tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_outputs = self.model(tensor)
        target_scaler = self.scalers.get(self.target_column)
        predictions: Dict[str, float] = {}
        for layer_name, tensor_out in raw_outputs.items():
            value = tensor_out.detach().cpu().numpy().reshape(-1, 1)
            if target_scaler is not None:
                unscaled = target_scaler.inverse_transform(value)[0, 0]
            else:
                unscaled = float(value[0, 0])
            alias = self.OUTPUT_ALIASES.get(layer_name, layer_name)
            predictions[alias] = float(unscaled)
        return pd.Series(predictions)

    def _prepare_sequence(self, symbol: str) -> np.ndarray:
        df = self._download_binance_candles(symbol)
        df = self._apply_preprocessing(df)
        features = df[self.feature_columns].dropna()
        if len(features) < self.sequence_length:
            raise RuntimeError(
                "No hay suficientes observaciones después del preprocesamiento para formar la secuencia."
            )
        window = features.iloc[-self.sequence_length :]
        return window.to_numpy(dtype=np.float32)

    def _download_binance_candles(self, symbol: str) -> pd.DataFrame:
        market_symbol = self._resolve_market_symbol(symbol)
        params = {
            "symbol": market_symbol,
            "interval": self.config.interval or "1d",
            "limit": max(self.sequence_length * 5, 200),
        }
        LOGGER.info(
            "Descargando velas de Binance para %s (intervalo %s)",
            market_symbol,
            params["interval"],
        )
        response = requests.get(BINANCE_API_URL, params=params, timeout=20)
        response.raise_for_status()
        raw = response.json()
        if not raw:
            raise RuntimeError(f"Binance no devolvió datos para {market_symbol}.")
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(raw, columns=columns)
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_base_volume",
        ]
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        df["volume_buy"] = df["taker_buy_base_volume"]
        df["volume_sell"] = df["volume"] - df["volume_buy"]
        df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        for column in self.log_columns:
            if column in processed:
                processed[column] = np.log1p(processed[column].clip(lower=0.0))
        for column in self.feature_columns:
            scaler = self.scalers.get(column)
            if scaler is None:
                continue
            values = processed[[column]].values
            processed[column] = scaler.transform(values)
        return processed

    def _resolve_market_symbol(self, symbol: str) -> str:
        if self.config.market_symbol:
            return self.config.market_symbol
        candidate = symbol.replace("-", "").replace("/", "").upper()
        if candidate.endswith("USD") and not candidate.endswith("USDT"):
            candidate = f"{candidate}T"
        return candidate
