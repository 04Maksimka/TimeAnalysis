from __future__ import annotations

import numpy as np
import pandas as pd

from time_analysis.backtesting import (
    SignalBacktestConfig,
    compare_signal_models,
    compare_signal_models_across_pairs,
    evaluate_signal_model,
)
from time_analysis.models import (
    BuyAndHoldBenchmarkModel,
    EmaTrendModel,
    SmaMomentumModel,
)


def test_evaluate_signal_model_returns_metrics_and_tables() -> None:
    """Check that one signal model backtest returns metrics and dataframes."""

    candles = _sample_candles(160)
    model = SmaMomentumModel(fast_window=3, slow_window=10)

    result = evaluate_signal_model(candles, model, SignalBacktestConfig())

    assert result.model_name == "SmaMomentumModel"
    assert not result.equity_curve.empty
    assert "total_return" in result.metrics
    assert "position" in result.equity_curve


def test_compare_signal_models_returns_sorted_metrics() -> None:
    """Check that model comparison returns one row per model."""

    candles = _sample_candles(160)
    models = [
        SmaMomentumModel(fast_window=3, slow_window=10),
        EmaTrendModel(fast_window=3, slow_window=10),
    ]

    metrics, results = compare_signal_models(candles, models)

    assert len(metrics) == 2
    assert set(results) == {"SmaMomentumModel", "EmaTrendModel"}
    assert list(metrics.columns).count("model") == 1


def test_compare_signal_models_across_pairs_returns_aggregate_metrics() -> None:
    """Check that multi-pair comparison returns aggregate model metrics."""

    candles_by_pair = {
        "UP/USDT": _sample_candles(160),
        "DOWN/USDT": _sample_candles(160),
    }
    candles_by_pair["DOWN/USDT"]["close"] = (
        candles_by_pair["DOWN/USDT"]["close"].iloc[::-1].to_numpy()
    )
    models = [
        BuyAndHoldBenchmarkModel(),
        SmaMomentumModel(fast_window=3, slow_window=10),
    ]

    pair_metrics, aggregate_metrics, results_by_pair = (
        compare_signal_models_across_pairs(
            candles_by_pair,
            models,
        )
    )

    assert len(pair_metrics) == 4
    assert len(aggregate_metrics) == 2
    assert set(results_by_pair) == {"UP/USDT", "DOWN/USDT"}
    assert "average_return" in aggregate_metrics


def test_evaluate_signal_model_can_use_short_positions() -> None:
    """Check that optional short signals are supported by the evaluator."""

    candles = _sample_candles(160)
    candles["close"] = candles["close"].iloc[::-1].to_numpy()
    model = BuyAndHoldBenchmarkModel()

    long_only = evaluate_signal_model(
        candles,
        model,
        SignalBacktestConfig(allow_short=False),
    )
    long_short = evaluate_signal_model(
        candles,
        model,
        SignalBacktestConfig(allow_short=True),
    )

    assert long_only.metrics["total_return"] == long_short.metrics["total_return"]


def _sample_candles(rows: int) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV candles for backtester tests.

    :param rows: number of rows to generate
    :return: OHLCV dataframe with a datetime index
    """

    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    trend = np.linspace(100.0, 120.0, rows)
    cycle = np.sin(np.arange(rows) / 8.0) * 1.5
    close = pd.Series(trend + cycle, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
    volume = pd.Series(1000.0 + np.sin(np.arange(rows) / 11.0) * 50.0, index=index)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
