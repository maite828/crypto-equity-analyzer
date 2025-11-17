import numpy as np
import pandas as pd

import scripts.merge_history as merge_history


def test_attach_macro_series_adds_enriched_features(monkeypatch):
    base = pd.DataFrame(
        {
            "close_time": pd.date_range("2024-01-01", periods=30, freq="H"),
            "close": np.linspace(100, 120, 30),
        }
    )
    base["return_1"] = base["close"].pct_change().fillna(0.0)

    macro = pd.DataFrame(
        {
            "close_time": base["close_time"],
            "macro_TEST_close": np.linspace(90, 110, 30),
        }
    )

    def fake_fetch(symbol, interval, days):
        return macro

    monkeypatch.setattr(merge_history, "fetch_macro_series", fake_fetch)

    enriched = merge_history.attach_macro_series(base.copy(), ["TEST"], "1h", 30)
    expected_cols = [
        "macro_TEST_return_1",
        "macro_TEST_return_5",
        "macro_TEST_volatility_20",
        "macro_TEST_corr_20",
        "macro_TEST_spread",
        "macro_TEST_beta_20",
    ]
    for col in expected_cols:
        assert col in enriched.columns
        assert np.isfinite(enriched[col]).all()
