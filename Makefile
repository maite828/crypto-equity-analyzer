# Secuencia lógica del pipeline
.PHONY: pipeline fetch merge train btc-local eth-local sol-local bnb-local xrp-local bch-local \
	docker-build clean-datasets clean-models clean-all dashboard docker-dashboard add-crypto remove-crypto

DEFAULT_SYMBOLS := BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT BCHUSDT
DEFAULT_COINS := bitcoin ethereum solana binancecoin ripple bitcoin-cash
SYMBOLS := $(shell python3 scripts/manage_assets.py list-symbols 2>/dev/null || echo $(DEFAULT_SYMBOLS))
COIN_IDS := $(shell python3 scripts/manage_assets.py list-coingecko 2>/dev/null || echo $(DEFAULT_COINS))
INTERVAL := 15m
DAYS := 365
ASSET_FAMILY := crypto
NEW_SYMBOL ?=
NEW_TICKER ?=
NEW_DISPLAY ?=
NEW_CG_ID ?=
NEW_TEXT_ONE ?=
NEW_TEXT_TWO ?=
ENABLE_DEFAULT ?= false
RANKING_ENABLED ?=
OVERWRITE_CONFIGS ?= false
REMOVE_SYMBOL ?=

DOCKER_PY := docker compose run --rm --entrypoint python analyzer

pipeline:
	$(DOCKER_PY) scripts/run_pipeline.py --symbols $(SYMBOLS) --coin-ids $(COIN_IDS) \
	  --intervals $(INTERVAL) --days $(DAYS) --asset-family $(ASSET_FAMILY)

fetch:
	$(DOCKER_PY) scripts/run_pipeline.py --symbols $(SYMBOLS) --coin-ids $(COIN_IDS) \
	  --intervals $(INTERVAL) --days $(DAYS) --skip-merge --skip-train --asset-family $(ASSET_FAMILY)

merge:
	$(DOCKER_PY) scripts/run_pipeline.py --symbols $(SYMBOLS) --coin-ids $(COIN_IDS) \
	  --intervals $(INTERVAL) --skip-fetch --skip-train --asset-family $(ASSET_FAMILY)

train:
	$(DOCKER_PY) scripts/run_pipeline.py --symbols $(SYMBOLS) --coin-ids $(COIN_IDS) \
	  --intervals $(INTERVAL) --skip-fetch --skip-merge --asset-family $(ASSET_FAMILY)

btc-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-btc-$(INTERVAL).yaml --skip-sentiment

eth-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-eth-$(INTERVAL).yaml --skip-sentiment

sol-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-sol-$(INTERVAL).yaml --skip-sentiment

bnb-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-bnb-$(INTERVAL).yaml --skip-sentiment

xrp-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-xrp-$(INTERVAL).yaml --skip-sentiment

bch-local:
	$(DOCKER_PY) -m app.cli --config configs/local-model-bch-$(INTERVAL).yaml --skip-sentiment

docker-build:
	docker compose build --no-cache analyzer pipeline

clean-datasets:
	rm -f data/datasets/*.parquet

clean-models:
	rm -f models/local_price_*$(INTERVAL)*.joblib reports/training_report_*$(INTERVAL)*.json

clean-all: clean-datasets clean-models

dashboard:
	streamlit run dashboard/app.py

docker-dashboard:
	@echo "Dashboard URL: http://localhost:8501"
	docker compose run --rm -p 8501:8501 --entrypoint bash analyzer -lc "cd /app && streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501" || true

add-crypto:
	@if [ -z "$(NEW_SYMBOL)" ] || [ -z "$(NEW_TICKER)" ] || [ -z "$(NEW_DISPLAY)" ] || [ -z "$(NEW_CG_ID)" ]; then \
		echo "Uso: make add-crypto NEW_SYMBOL=UNIUSDT NEW_TICKER=UNI NEW_DISPLAY=\"Uniswap\" NEW_CG_ID=uniswap [ENABLE_DEFAULT=true]"; \
		exit 1; \
	fi
	@$(DOCKER_PY) scripts/manage_assets.py add-crypto \
		--symbol $(NEW_SYMBOL) \
		--ticker $(NEW_TICKER) \
		--display "$(NEW_DISPLAY)" \
		--coingecko-id $(NEW_CG_ID) \
		$(if $(filter true,$(ENABLE_DEFAULT)),--enable-default,) \
		$(if $(RANKING_ENABLED),--ranking-enabled $(RANKING_ENABLED),) \
		$(if $(NEW_TEXT_ONE),--text-one "$(NEW_TEXT_ONE)",) \
		$(if $(NEW_TEXT_TWO),--text-two "$(NEW_TEXT_TWO)",) \
		$(if $(filter true,$(OVERWRITE_CONFIGS)),--overwrite-configs,)

remove-crypto:
	@if [ -z "$(REMOVE_SYMBOL)" ]; then \
		echo "Uso: make remove-crypto REMOVE_SYMBOL=KSMUSDT"; \
		exit 1; \
	fi
	@$(DOCKER_PY) scripts/manage_assets.py remove-crypto --symbol $(REMOVE_SYMBOL)
