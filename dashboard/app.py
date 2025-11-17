import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import load_config
from app.sentiment import SentimentService
from app.data_sources.rss import fetch_feed
from app.assets import (
    ALL_CRYPTO_SYMBOLS,
    CRYPTO_RANKING_SYMBOLS,
    CRYPTO_TICKERS,
)

BASE_DIR = ROOT_DIR
DEFAULT_CONFIG = ROOT_DIR / "configs" / "sample-config.yaml"
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
CRYPTO_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "The Block": "https://www.theblock.co/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Reuters Crypto": "https://feeds.reuters.com/reuters/technologyNews",
}

EQUITY_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
    "Financial Times": "https://www.ft.com/companies?format=rss",
}
INTERVAL_OPTIONS = ["5m", "15m", "1h", "1d"]
DEFAULT_INTERVAL = "15m"
FAMILY_SYMBOLS = {
    "crypto": ALL_CRYPTO_SYMBOLS,
    "equity": ["TSLA", "NIO", "NVDA", "AMZN", "GOOGL"],
}
FAMILY_RANKING_SYMBOLS = {
    "crypto": CRYPTO_RANKING_SYMBOLS,
    "equity": ["TSLA", "NIO", "NVDA", "AMZN", "GOOGL"],
}
FAMILY_INTERVALS = {
    "crypto": ["5m", "15m", "1h"],
    "equity": ["1h", "1d"],
}
LOCAL_CONFIG_MAP = {
    "crypto": {ticker: ticker.lower() for ticker in CRYPTO_TICKERS},
    "equity": {
        "TSLA": "tsla",
        "NIO": "nio",
        "NVDA": "nvda",
        "AMZN": "amzn",
        "GOOGL": "googl",
    },
}
CONFIG_DIR = BASE_DIR / "configs"


def fetch_real_prices(symbols):
    real_prices = {}
    for symbol in symbols:
        try:
            hist = yf.download(symbol, period="5d", interval="1d", auto_adjust=False, progress=False)
        except Exception as exc:  # pragma: no cover
            st.warning(f"No se pudo obtener precio real para {symbol}: {exc}")
            continue
        if hist.empty:
            continue
        close_series = hist["Close"].iloc[-1]
        try:
            real_prices[symbol] = float(close_series)
        except TypeError:
            real_prices[symbol] = float(close_series.iloc[0])
    return real_prices


def run_command(args):
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    base = str(BASE_DIR)
    if pythonpath:
        env["PYTHONPATH"] = f"{base}:{pythonpath}"
    else:
        env["PYTHONPATH"] = base
    process = subprocess.run(
        ["python3"] + args,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n{process.stdout}\n{process.stderr}"
        )
    return process.stdout + process.stderr


def latest_dataset(symbol: str, interval: str, asset_family: str) -> str:
    pattern = DATASETS_DIR.glob(f"{symbol}_{asset_family}_{interval}_*rows.parquet")
    files = list(pattern)
    if not files:
        raise RuntimeError(
            f"No se encontró dataset para {symbol} ({interval}, {asset_family}). Ejecuta Merge/Pipeline primero."
        )
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def run_pipeline_cli(
    symbols,
    intervals,
    days,
    asset_family,
    model_type,
    skip_fetch=False,
    skip_merge=False,
    skip_train=False,
):
    if not symbols:
        raise ValueError("Selecciona al menos un símbolo para ejecutar el pipeline.")
    if not intervals:
        raise ValueError("Selecciona al menos un intervalo.")
    args = [
        "scripts/run_pipeline.py",
        "--symbols",
        *symbols,
        "--intervals",
        *intervals,
        "--days",
        str(days),
        "--asset-family",
        asset_family,
        "--model-type",
        model_type,
    ]
    if skip_fetch:
        args.append("--skip-fetch")
    if skip_merge:
        args.append("--skip-merge")
    if skip_train:
        args.append("--skip-train")
    return run_command(args)


def run_tuning_cli(symbols, intervals, asset_family, model_type, n_iter):
    outputs = []
    for symbol in symbols:
        for interval in intervals:
            dataset_path = latest_dataset(symbol, interval, asset_family)
            args = [
                "scripts/tune_model.py",
                "--symbol",
                symbol,
                "--interval",
                interval,
                "--asset-family",
                asset_family,
                "--model-type",
                model_type,
                "--n-iter",
                str(n_iter),
                "--dataset",
                dataset_path,
            ]
            outputs.append(run_command(args))
    return "\n".join(outputs)


def get_latest_ranking(symbols):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "python3",
                "scripts/batch_analysis.py",
                "--symbols",
                *symbols,
                "--skip-sentiment",
                "--output",
                tmp_path,
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        df = pd.read_csv(tmp_path, comment="#")
        summary = None
        with open(tmp_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("# Interpretación"):
                    summary = line.split(":", 1)[1].strip()
        return df, summary
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_equity_ranking(symbols, interval="1h"):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    summary_path = tmp_path + ".txt"
    try:
        result = subprocess.run(
            [
                "python3",
                "scripts/batch_equity_ranking.py",
                "--symbols",
                *symbols,
                "--interval",
                interval,
                "--output",
                tmp_path,
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        df = pd.read_csv(tmp_path)
        summary = Path(summary_path).read_text(encoding="utf-8").strip() if os.path.exists(summary_path) else None
        return df, summary
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(summary_path):
            os.remove(summary_path)


def load_local_prediction(config_path):
    result = subprocess.run(
        [
            "python3",
            "-m",
            "app.cli",
            "--config",
            str(config_path),
            "--skip-sentiment",
        ],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def main():
    st.title("Crypto Prediction Dashboard")

    asset_family = st.selectbox(
        "Familia de activos",
        options=list(FAMILY_SYMBOLS.keys()),
        index=0,
        help="Selecciona si quieres operar con cripto (Binance) o acciones (Yahoo Finance).",
    )
    available_symbols = FAMILY_SYMBOLS[asset_family]
    family_intervals = FAMILY_INTERVALS.get(asset_family, INTERVAL_OPTIONS)
    interval_choices = family_intervals or INTERVAL_OPTIONS

    with st.expander("Orquestador (equivalente a make)"):
        max_days = 730 if asset_family == "equity" else 365
        default_days = 730 if asset_family == "equity" else 30
        days = st.slider(
            "Historic days",
            min_value=1,
            max_value=max_days,
            value=min(default_days, max_days),
            help="Profundidad de backfill para fetch.",
        )
        selected_symbols = st.multiselect(
            "Símbolos",
            available_symbols,
            default=available_symbols,
        )
        default_interval_list = [interval_choices[0]] if interval_choices else [DEFAULT_INTERVAL]
        selected_intervals = st.multiselect(
            "Intervalos",
            interval_choices,
            default=default_interval_list,
            help="Puedes seleccionar varios para entrenar múltiples horizontes.",
        )
        model_type = st.selectbox(
            "Modelo",
            ["random_forest", "xgboost", "lightgbm"],
            help="Selecciona el estimador para la fase de entrenamiento.",
        )
        tune_iterations = st.slider(
            "Iteraciones tuning (RandomizedSearch)",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            help="Número de combinaciones a probar cuando ejecutes Tune.",
        )
        col_fetch, col_merge, col_train, col_pipe, col_tune = st.columns(5)
        if col_fetch.button("Fetch"):
            with st.spinner("Descargando datos públicos..."):
                try:
                    output = run_pipeline_cli(
                        selected_symbols,
                        selected_intervals,
                        days,
                        asset_family,
                        model_type,
                        skip_merge=True,
                        skip_train=True,
                    )
                    st.code(output or "Fetch completado.", language="bash")
                except Exception as exc:
                    st.error(str(exc))
        if col_merge.button("Merge"):
            with st.spinner("Construyendo datasets..."):
                try:
                    output = run_pipeline_cli(
                        selected_symbols,
                        selected_intervals,
                        days,
                        asset_family,
                        model_type,
                        skip_fetch=True,
                        skip_train=True,
                    )
                    st.code(output or "Merge completado.", language="bash")
                except Exception as exc:
                    st.error(str(exc))
        if col_train.button("Train"):
            with st.spinner("Entrenando modelos locales..."):
                try:
                    output = run_pipeline_cli(
                        selected_symbols,
                        selected_intervals,
                        days,
                        asset_family,
                        model_type,
                        skip_fetch=True,
                        skip_merge=True,
                    )
                    st.code(output or "Entrenamiento completado.", language="bash")
                except Exception as exc:
                    st.error(str(exc))
        if col_pipe.button("Pipeline completo"):
            with st.spinner("Ejecutando fetch→merge→train..."):
                try:
                    output = run_pipeline_cli(
                        selected_symbols,
                        selected_intervals,
                        days,
                        asset_family,
                        model_type,
                    )
                    st.code(output or "Pipeline completado.", language="bash")
                except Exception as exc:
                    st.error(str(exc))
        if col_tune.button("Tune models"):
            with st.spinner("Buscando hiperparámetros óptimos..."):
                try:
                    output = run_tuning_cli(
                        selected_symbols,
                        selected_intervals,
                        asset_family,
                        model_type,
                        tune_iterations,
                    )
                    st.code(output or "Tuning completado.", language="bash")
                except Exception as exc:
                    st.error(str(exc))

    st.header("Local Predictions")
    interval_index = 0
    if DEFAULT_INTERVAL in interval_choices:
        interval_index = interval_choices.index(DEFAULT_INTERVAL)
    pred_interval = st.selectbox(
        "Intervalo para los modelos locales",
        interval_choices,
        index=min(interval_index, max(len(interval_choices) - 1, 0)),
    )
    family_configs = LOCAL_CONFIG_MAP.get(asset_family, {})
    cols = st.columns(len(family_configs) or 1)
    for col, (symbol, prefix) in zip(cols, family_configs.items()):
        config_path = CONFIG_DIR / f"local-model-{prefix}-{pred_interval}.yaml"
        if col.button(f"Predict {symbol} ({pred_interval})"):
            if not config_path.exists():
                col.warning("Modelo no disponible. Ejecuta el pipeline primero.")
            else:
                output = load_local_prediction(config_path)
                col.text(output)

    st.header("Multi-symbol Ranking")
    if asset_family == "crypto":
        rank_symbols = st.multiselect(
            "Activos a rankear",
            FAMILY_RANKING_SYMBOLS["crypto"],
            default=FAMILY_RANKING_SYMBOLS["crypto"],
        )
        if st.button("Run Batch Analysis"):
            if not rank_symbols:
                st.warning("Selecciona al menos un activo para generar el ranking.")
            else:
                try:
                    df, summary = get_latest_ranking(rank_symbols)
                    if df.empty:
                        st.warning("No se obtuvo ranking (verifica que existan modelos para esos símbolos).")
                    else:
                        enriched = df.copy()
                        real_prices = fetch_real_prices(enriched["asset"].unique())
                        if real_prices:
                            enriched["real_price"] = enriched["asset"].map(real_prices)
                            for horizon in ("1d", "7d"):
                                exp_col = f"expected_price_{horizon}"
                                real_exp_col = f"real_expected_{horizon}"
                                if exp_col in enriched.columns:
                                    enriched[real_exp_col] = enriched.apply(
                                        lambda row: row["real_price"] / row["current_price"] * row[exp_col]
                                        if row.get("real_price") and row["current_price"]
                                        else None,
                                        axis=1,
                                    )
                        else:
                            enriched["real_price"] = None
                        for horizon in ("1d", "7d"):
                            delta_col = f"delta_{horizon}"
                            pct_col = f"{delta_col}_pct"
                            if delta_col in enriched.columns:
                                enriched[pct_col] = (
                                    enriched[delta_col] / enriched["current_price"]
                                ) * 100
                        display_cols = [
                            "asset",
                            "current_price",
                            "real_price",
                            "expected_price_1d",
                            "real_expected_1d",
                            "expected_price_7d",
                            "real_expected_7d",
                            "delta_1d",
                            "delta_1d_pct",
                            "delta_7d",
                            "delta_7d_pct",
                        ]
                        available_cols = [col for col in display_cols if col in enriched.columns]
                        st.dataframe(
                            enriched[available_cols].rename(
                                columns={
                                    "asset": "Símbolo",
                                    "current_price": "Precio modelo",
                                    "real_price": "Precio real",
                                    "expected_price_1d": "Precio modelo (1d)",
                                    "real_expected_1d": "Precio real (1d)",
                                    "expected_price_7d": "Precio modelo (7d)",
                                    "real_expected_7d": "Precio real (7d)",
                                    "delta_1d": "Δ 1d (USD)",
                                    "delta_1d_pct": "Δ 1d (%)",
                                    "delta_7d": "Δ 7d (USD)",
                                    "delta_7d_pct": "Δ 7d (%)",
                                }
                            ),
                            width="stretch",
                        )
                    if summary:
                        st.info(summary)
                except RuntimeError as e:
                    st.error(str(e))
    else:
        equity_symbols = st.multiselect(
            "Acciones a rankear",
            FAMILY_RANKING_SYMBOLS["equity"],
            default=FAMILY_RANKING_SYMBOLS["equity"],
        )
        if st.button("Run Equity Ranking"):
            if not equity_symbols:
                st.warning("Selecciona al menos una acción.")
            else:
                try:
                    df, summary = get_equity_ranking(equity_symbols, interval=pred_interval)
                    if df.empty:
                        st.warning("No se generó ranking (¿faltan modelos locales?).")
                    else:
                        st.dataframe(
                            df.rename(
                                columns={
                                    "asset": "Símbolo",
                                    "current_price": "Precio actual",
                                    "expected_price_1d": "Precio esperado (modelo)",
                                    "delta_1d": "Δ (USD)",
                                    "delta_1d_pct": "Δ (%)",
                                    "sentiment": "Sentimiento",
                                    "sentiment_confidence": "Confianza sentimiento",
                                    "benchmark_return_1": "Retorno benchmark",
                                    "days_to_next_earnings": "Días a earnings",
                                    "has_upcoming_earnings": "Próximos earnings",
                                }
                            ),
                            width="stretch",
                        )
                    if summary:
                        st.info(summary)
                except RuntimeError as exc:
                    st.error(str(exc))

    st.header("Latest Sentiment")
    feeds = CRYPTO_FEEDS if asset_family == "crypto" else EQUITY_FEEDS
    feed_choice = st.selectbox(
        "Fuente",
        list(feeds.keys()) + ["Personalizada"],
        key=f"feed_choice_{asset_family}",
    )
    default_url = list(feeds.values())[0] if feeds else "https://www.coindesk.com/arc/outboundfeeds/rss/"
    custom_url = st.text_input(
        "RSS personalizado",
        value=default_url,
        help="Sólo se usa si eliges 'Personalizada'.",
        key=f"custom_feed_{asset_family}",
    )
    feed_url = custom_url if feed_choice == "Personalizada" else feeds[feed_choice]
    limit = st.slider("Número de titulares", 3, 10, 5)
    if st.button("Analizar sentimiento"):
        try:
            feed = fetch_feed(feed_url)
            if feed.empty:
                st.warning("El feed no devolvió titulares.")
            else:
                texts = feed["title"].head(limit).tolist()
                config = load_config(DEFAULT_CONFIG)
                service = SentimentService(config.sentiment)
                df = service.analyze(texts)
                st.dataframe(df)
        except Exception as exc:
            st.error(f"No se pudo analizar el sentimiento: {exc}")


if __name__ == "__main__":
    main()
