# Guía de uso · Criptomonedas

Esta guía cubre el flujo completo para descargar datos de Binance/CoinGecko, fusionarlos y entrenar modelos locales para BTC, ETH, SOL, BNB, XRP, BCH, KTA, UNI y KSM. Se ejecuta todo en Docker, por lo que no necesitas dependencias adicionales.

## Pasos rápidos con Make + Docker

```bash
make docker-build                                # reconstruye la imagen tras cambios
make pipeline INTERVAL=15m                       # fetch + merge + train para todos los símbolos cripto
make pipeline INTERVAL="5m 1h" DAYS=180          # procesa dos granularidades con 180 días históricos
make btc-local INTERVAL=5m                       # ejecuta python -m app.cli con configs/local-model-btc-5m.yaml
make docker-dashboard                            # abre http://localhost:8501 con Streamlit
```

Variables relevantes:

- `INTERVAL`: `5m`, `15m`, `1h` (puedes pasar múltiples valores entre comillas).
- `SYMBOLS`: limita los pares, ej. `make pipeline SYMBOLS="BTCUSDT ETHUSDT"`.
- `DAYS`: días a descargar con `fetch_public_data`.

## Desde el dashboard

1. Lanza `make docker-dashboard`.
2. Selecciona la familia **crypto** en la parte superior.
3. En el expander *Orquestador* ajusta días, símbolos e intervalos:
   - `Fetch`: sólo Binance + CoinGecko + RSS + on-chain.
   - `Merge`: construye datasets en `data/datasets/`.
   - `Train`: entrena modelos locales en `models/local_price_{symbol}_{interval}.joblib`.
   - `Pipeline completo`: encadena las tres fases.
4. *Local Predictions*: elige intervalo y pulsa el botón del símbolo para ver la predicción (usa la config `configs/local-model-{symbol}-{interval}.yaml`).
5. *Multi-symbol Ranking*: ejecuta `scripts/batch_analysis.py` para BTC/ETH/SOL/BNB/XRP/BCH/KTA/UNI/KSM; el dashboard calcula deltas en USD y % y muestra la **interpretación automática**.
6. *Latest Sentiment*: analiza titulares reales desde el feed RSS seleccionado usando `cardiffnlp/twitter-roberta-base-sentiment-latest`.

## Gestión de activos

- El inventario cripto está centralizado en `assets/crypto_assets.yaml`. Cada entrada indica símbolo Binance (`symbol`), ticker (`ticker`), CoinGecko ID (`coin_id`), nombre descriptivo y textos por defecto.
- Para añadir una moneda sin editar a mano:  
  ```bash
  make add-crypto NEW_SYMBOL=UNIUSDT NEW_TICKER=UNI NEW_DISPLAY="Uniswap" NEW_CG_ID=uniswap ENABLE_DEFAULT=true
  ```  
  Esto actualiza el YAML y genera automáticamente `configs/local-model-uni-{5m,15m,1h}.yaml`. Puedes pasar `NEW_TEXT_ONE`, `NEW_TEXT_TWO`, `RANKING_ENABLED=true|false` y `OVERWRITE_CONFIGS=true` si necesitas personalizar los textos o reemplazar configs existentes.
- El listado por defecto (`make pipeline` sin parámetros) contiene los símbolos con `enabled: true`. Cambia ese valor en el YAML (o usa `--enable-default`) para incluir/excluir monedas del pipeline base.
- Para borrar una moneda y todos sus artefactos (datasets, modelos, configs): `make remove-crypto REMOVE_SYMBOL=KSMUSDT`.

## Detalle de comandos

### Ingesta (`scripts/fetch_public_data.py`)

- Binance OHLCV por intervalo (`data/market/binance_{symbol}_{interval}_YYYYMM.parquet`).
- Order book (`data/liquidity/orderbook_{symbol}_{interval}_*.parquet`).
- CoinGecko markets (`data/spot/`), status updates (`data/text/`).
- RSS (`data/news/`), on-chain (Blockchain.com) en `data/onchain/`.
- Fuentes opcionales: Twitter, Reddit, CoinMarketCal, Glassnode, CoinMetrics.

### Merge (`scripts/merge_history.py`)

Combina:

- Indicadores técnicos (EMA, RSI, ATR, retornos, volatilidades).
- Desequilibrio del order book (si `--include-order-book`).
- Métricas spot de CoinGecko y on-chain (si `--include-spot`, `--include-onchain`).

El resultado se guarda en `data/datasets/{symbol}_{interval}_{rows}rows.parquet`.

### Entrenamiento (`scripts/train_price_model.py`)

Construye features (`target_price`, `target_return`), divide en train/test y entrena un `RandomForestRegressor`. Guarda:

- Modelo: `models/local_price_{symbol}_{interval}.joblib`.
- Reporte: `reports/training_report_{symbol}_{interval}.json` (MAE, RMSE, filas, features).

### CLI (`python -m app.cli --config ...`)

- Lee la config (ej. `configs/local-model-btc-15m.yaml`), carga el modelo local y genera:
  ```
  Predicción de precios para BTC-USD:
  predicted_price    102489.91
  horizon_steps           28.00
  ```
- Con `scripts/batch_analysis.py` puedes analizar varios símbolos y obtener un CSV (`outputs/ranking.csv`) con la interpretación automática.

## Flujo recomendado antes de invertir

1. Ejecuta `make pipeline` (o el botón equivalente en el dashboard) para garantizar que los datos y modelos están actualizados.
2. Revisa *Latest Sentiment* con un feed fiable.
3. Analiza el ranking multi-símbolo (fíjate en `delta_7d` y `delta_1d`).
4. Confirma con la predicción individual en *Local Predictions*.
5. Si todo se alinea (sentimiento + ranking + predicción), tendrás una señal más robusta.

## Configs disponibles

- `configs/local-model-{symbol}-{interval}.yaml`: modelos locales de 5m/15m/1h para BTC, ETH, SOL, BNB, XRP, BCH, KTA, UNI, KSM.
- `configs/sample-config.yaml`: ejemplo para el modelo público de Hugging Face.
- `configs/crypto-hf-template.yaml`: plantilla para tu propio repositorio en Hugging Face Hub.

## Automatización

Programa el pipeline diario (ejemplo crontab):

```
0 3 * * * cd /ruta/al/repo && docker compose run --rm pipeline >> logs/pipeline.log 2>&1
```

El dashboard (`make docker-dashboard`) te permite refrescar datos y consultar señales sin salir del navegador.
