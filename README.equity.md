# Guía de uso · Acciones (Equity)

El pipeline de equities reutiliza la misma infraestructura del proyecto, pero cambia las fuentes: velas de Yahoo Finance, sin order book ni métricas on-chain. Aquí tienes los pasos para trabajar con tickers como TSLA, NIO, NVDA, AMZN y GOOGL.

## Variables clave

- `ASSET_FAMILY=equity` (obligatoria para todas las fases).
- `SYMBOLS="TSLA NVDA ..."` (toma tickers reales de mercado).
- `INTERVAL`: actualmente soportado `1h` (puedes usar `1d` si necesitas históricos diarios).
- `DAYS`: cantidad de días a descargar vía yfinance (ej. 120, 365).

## Comandos principales

```bash
make docker-build
make pipeline ASSET_FAMILY=equity SYMBOLS="TSLA NVDA" INTERVAL=1h DAYS=365 MODEL_TYPE=xgboost

# Paso a paso, si prefieres separarlo:
make fetch ASSET_FAMILY=equity SYMBOLS="TSLA NVDA" INTERVAL=1h DAYS=180
make merge ASSET_FAMILY=equity SYMBOLS="TSLA NVDA" INTERVAL=1h
make train ASSET_FAMILY=equity SYMBOLS="TSLA NVDA" INTERVAL=1h MODEL_TYPE=random_forest
```

Las velas se guardan en `data/market_equities/equity_{symbol}_{interval}_YYYYMM.parquet` y los datasets resultantes en `data/datasets/{symbol}_equity_{interval}_{rows}rows.parquet`.


## Ranking multi-símbolo

- Desde el dashboard (familia **equity**) pulsa **Run Equity Ranking** para mostrar el ranking ordenado por `Δ (USD)` y `Δ (%)`. El panel llama a `scripts/batch_equity_ranking.py` usando los modelos locales entrenados.
- También puedes ejecutarlo manualmente:
  ```bash
  python scripts/batch_equity_ranking.py \
    --symbols TSLA NIO NVDA AMZN GOOGL \
    --interval 1h \
    --output outputs/equity-ranking.csv
  ```
  El CSV resultante incluye las columnas `current_price`, `expected_price_1d`, `delta_1d`, `delta_1d_pct`, `sentiment` y `sentiment_confidence`, junto con un resumen en `outputs/equity-ranking.csv.txt`.

## Configuraciones de ejemplo

Revisa los YAML añadidos en `configs/`:

- `configs/local-model-tsla-1h.yaml`
- `configs/local-model-nio-1h.yaml`
- `configs/local-model-nvda-1h.yaml`
- `configs/local-model-amzn-1h.yaml`
- `configs/local-model-googl-1h.yaml`

Cada uno apunta a un modelo local en `models/local_price_{TICKER}_equity_1h.joblib` y define `asset_family: equity`. Tras entrenar, ejecuta:

```bash
python -m app.cli --config configs/local-model-tsla-1h.yaml --skip-sentiment
```
- Estos YAML ya usan FinBERT (`ProsusAI/finbert`) para el análisis de noticias y fijan `benchmark_symbol: "^GSPC"` para generar features relativos al S&P 500.

## Features avanzadas

El merge y la construcción de features incorporan:

- Indicadores técnicos adicionales: OBV, VWAP y ratio VWAP, además de EMA/RSI estándar.
- Relación con el benchmark (`^GSPC` por defecto): `benchmark_close`, `benchmark_return_1`, `close_to_benchmark`.
- Señales corporativas: `days_to_next_earnings`, `has_upcoming_earnings`, `last_dividend`, `dividend_yield` (extraídas de yfinance).
- Modelos entrenados con más árboles/profundidad cuando `ASSET_FAMILY=equity` (`RandomForest` con 400 árboles y profundidad 10 por defecto) y validación walk-forward configurable (`--walk-splits`).
- Puedes alternar entre RandomForest y XGBoost estableciendo `MODEL_TYPE=xgboost` (o usando `--model-type` al llamar a `scripts/train_price_model.py`).

## Validación

- `scripts/train_price_model.py` calcula métricas de hold-out (MAE/RMSE) y de validación walk-forward (`--walk-splits`, por defecto 5) usando `TimeSeriesSplit`. Los resultados quedan registrados en `reports/training_report_{symbol}_equity_{interval}.json`.
- Puedes ajustar hiperparámetros por ticker con `--n-estimators`/`--max-depth` o modificando los YAML correspondientes.

## Dashboard

1. `make docker-dashboard` → abre `http://localhost:8501`.
2. Selecciona la familia **equity**.
3. Usa el expander *Orquestador* para lanzar Fetch/Merge/Train con tus tickers.
4. *Local Predictions* mostrará los botones para TSLA, NIO, NVDA, AMZN y GOOGL (intervalo 1h).
5. *Multi-symbol Ranking* ahora también muestra las acciones seleccionadas (usa el botón **Run Equity Ranking**) e incluye el sentimiento agregado de sus últimas noticias.
6. *Latest Sentiment* cambia las fuentes automáticamente (Yahoo Finance, Investing.com, Seeking Alpha, Financial Times) cuando seleccionas la familia equity.

## ¿Qué cambia frente a cripto?

- Fuente de velas: Yahoo Finance (`yfinance`) en lugar de Binance.
- No se descargan order books ni métricas on-chain (no aplican a equities).
- Los modelos y reportes incluyen el sufijo `_equity_` para evitar colisiones con los de cripto.
- Algunos parámetros (intervalos soportados, lookback) se ajustan a la disponibilidad de yfinance.

## Buenas prácticas

1. Define siempre `SYMBOLS` explícitamente para evitar descargar datos innecesarios.
2. Usa `DAYS=120` o más para tener suficiente historial antes de entrenar.
3. Revisa que los modelos (`models/local_price_{ticker}_equity_1h.joblib`) existan antes de lanzar predicciones.
4. Si necesitas otros intervalos (ej. `1d`), asegúrate de actualizar los YAML y volver a entrenar.

Con estos pasos puedes operar acciones y criptos desde el mismo proyecto, sin duplicar repositorios.
