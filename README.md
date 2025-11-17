# Analizador Genérico de Sentimiento y Precios

Este proyecto descarga datos públicos, genera datasets enriquecidos y entrena modelos locales (RandomForest) para predecir precios en dos familias de activos:

- **Cripto**: Binance + CoinGecko + métricas on-chain + RSS.
- **Equities**: Yahoo Finance (yfinance) para tickers como TSLA, NIO, NVDA, AMZN, GOOGL.

Todo se orquesta con scripts de Python, Make + Docker y un dashboard en Streamlit.

## Documentación específica

- [README.crypto.md](README.crypto.md): flujo detallado para criptomonedas (Binance, CoinGecko, order books, ranking multi-símbolo, etc.).
- [README.equity.md](README.equity.md): guía para acciones (pipelines con `ASSET_FAMILY=equity`, configs TSLA/NVDA/AMZN/GOOGL/NIO).

Consulta cada archivo para conocer las banderas, rutas y comandos exactos de cada familia.

## Resumen rápido

```
app/               # código de la CLI y servicios (sentiment, pricing, features)
configs/           # YAML para cada símbolo/intervalo (crypto y equity)
scripts/           # fetch_public_data, merge_history, train_price_model, run_pipeline
dashboard/app.py   # panel Streamlit (make docker-dashboard)
Makefile           # atajos make fetch/merge/train/pipeline/docker-build
data/, models/, outputs/, reports/
```

1. **Construye la imagen**: `make docker-build`.
2. **Ejecuta el pipeline**:
   ```bash
   make pipeline INTERVAL=15m                             # cripto (por defecto)
   make pipeline INTERVAL="5m 1h" SYMBOLS="BTCUSDT ETHUSDT"
   make pipeline ASSET_FAMILY=equity SYMBOLS="TSLA NVDA" INTERVAL=1h DAYS=180
   ```
   > En modo equity usa tickers reales (TSLA, NVDA, AMZN…). Los símbolos con sufijo `USDT` sólo funcionan en la familia crypto.
3. **Dashboard**: `make docker-dashboard` → http://localhost:8501 (elige familia, lanza pipeline, consulta predicciones).

Variables relevantes:

- `SYMBOLS`, `INTERVAL`, `DAYS`, `ASSET_FAMILY` (crypto | equity).
- `MODEL_TYPE` (opcional): `random_forest` (default), `xgboost` o `lightgbm` para usar el estimador deseado.
- En el dashboard puedes elegir el modelo desde el dropdown “Modelo” antes de pulsar Fetch/Merge/Train/Pipeline; aplica los mismos valores (`random_forest` o `xgboost`).
- Los modelos locales se guardan como `models/local_price_{symbol}_{family}_{interval}.joblib`.
- Los datasets enriquecidos terminan en `data/datasets/{symbol}_{family}_{interval}_{rows}rows.parquet`.

### Gestión de activos cripto

- El inventario vive en `assets/crypto_assets.yaml`. Ahí defines símbolo Binance, ticker, CoinGecko ID, textos por defecto y si participa en la lista por defecto.
- Usa `make add-crypto NEW_SYMBOL=UNIUSDT NEW_TICKER=UNI NEW_DISPLAY="Uniswap" NEW_CG_ID=uniswap [ENABLE_DEFAULT=true]` para añadir un activo. El comando actualiza el YAML y genera `configs/local-model-{ticker}-{interval}.yaml`.
- Para eliminar todo rastro de una moneda (entrada en el YAML, configs y artefactos): `make remove-crypto REMOVE_SYMBOL=KSMUSDT`.
- Si prefieres más control (o quieres sobreescribir los textos), ejecuta `python scripts/manage_assets.py add-crypto --help` para ver todas las opciones (`--text-one`, `--text-two`, `--ranking-enabled`, etc.).

## Comandos útiles

- `make fetch` / `make merge` / `make train`: ejecutan cada fase por separado (honran `ASSET_FAMILY` y `SYMBOLS`).
- `python -m app.cli --config configs/local-model-btc-15m.yaml --skip-sentiment` (cripto) o `configs/local-model-tsla-1h.yaml` (equity) para predicciones individuales.
- `scripts/batch_analysis.py` (usado desde el dashboard) imprime la tabla de ranking e interpretación automática.
- `scripts/batch_equity_ranking.py` genera el ranking multiactivo para acciones y calcula sentimiento con FinBERT (el dashboard lo llama cuando eliges la familia equity).
- `scripts/train_price_model.py` admite `--n-estimators`, `--max-depth` y `--walk-splits` para ajustar modelos y obtener métricas walk-forward (especialmente útil en equities).
- `scripts/tune_model.py` ejecuta RandomizedSearchCV sobre RF/XGBoost para un dataset concreto y guarda los mejores parámetros en `configs/model_params/`.
- `pytest`: ejecuta `python -m pytest` para validar los módulos (tests enfocados en features y ranking de equities).

> Si prefieres ejecutar sin Docker, instala `pip install -r requirements.txt` y usa los scripts directamente. Aun así, las instrucciones oficiales están pensadas para correr en contenedores.

## Automatización

Programa el pipeline diario con `cron`, por ejemplo:

```
0 3 * * * cd /ruta/al/repo && docker compose run --rm pipeline >> logs/pipeline.log 2>&1
```

Usa `make docker-dashboard` para revisar datos, ranking y sentimiento sin salir del navegador.
