"""Servicios relacionados al análisis de sentimiento."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from transformers import pipeline

from .config import SentimentSection

LOGGER = logging.getLogger("app.sentiment")


class SentimentService:
    """Envuelve un pipeline de Hugging Face para facilitar el switcheo de modelos."""

    def __init__(self, config: SentimentSection) -> None:
        self.config = config
        if not config.enabled:
            raise ValueError("El módulo de sentimiento está deshabilitado en la configuración.")
        if config.provider != "huggingface":
            raise NotImplementedError(
                f"Proveedor de sentimiento '{config.provider}' no implementado."
            )

        LOGGER.info("Cargando modelo de sentimiento %s", config.model_id)
        self._pipeline = pipeline(
            task=config.task,
            model=config.model_id,
            tokenizer=config.model_id,
            device=config.device,
        )

    def analyze(self, texts: Sequence[str]) -> pd.DataFrame:
        if not texts:
            raise ValueError("Debe proporcionar al menos un texto para analizar.")
        LOGGER.info("Analizando %d textos...", len(texts))
        predictions = self._pipeline(
            list(texts),
            batch_size=self.config.batch_size,
            truncation=True,
        )
        df = pd.DataFrame(predictions)
        df.insert(0, "text", texts)
        df.rename(columns={"label": "sentiment", "score": "confidence"}, inplace=True)
        return df
