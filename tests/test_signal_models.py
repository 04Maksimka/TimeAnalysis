from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_analysis.models import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    AtrVolatilityBreakoutModel,
    BollingerRsiMeanReversionModel,
    BuyAndHoldBenchmarkModel,
    DonchianBreakoutModel,
    EmaTrendModel,
    HistGradientBoostingDirectionModel,
    MacdRsiTrendModel,
    RandomForestDirectionModel,
    SignalModel,
    SmaMomentumModel,
    WarmupMomentumHoldModel,
    default_model_zoo,
)


def test_default_model_zoo_uses_shared_interface() -> None:
    """Check that default model candidates implement the shared interface."""

    models = default_model_zoo(include_ml=False)

    assert models
    assert all(isinstance(model, SignalModel) for model in models)


@pytest.mark.parametrize(
    "model",
    [
        SmaMomentumModel(fast_window=3, slow_window=8),
        BuyAndHoldBenchmarkModel(),
        WarmupMomentumHoldModel(lookback_candles=12, minimum_return=0.001),
        EmaTrendModel(fast_window=3, slow_window=8),
        DonchianBreakoutModel(entry_window=10, exit_window=5),
        BollingerRsiMeanReversionModel(window=10, rsi_window=5),
        MacdRsiTrendModel(fast_window=3, slow_window=8, signal_window=3),
        AtrVolatilityBreakoutModel(atr_window=5, ema_window=10),
    ],
)
def test_rule_based_models_emit_standard_signal_columns(model: SignalModel) -> None:
    """Check that rule-based models return standard signal columns.

    :param model: signal model fixture from pytest parameterization
    """

    candles = _sample_candles(120)

    result = model.predict(candles)

    assert ENTRY_SIGNAL_COLUMN in result
    assert EXIT_SIGNAL_COLUMN in result
    assert result[ENTRY_SIGNAL_COLUMN].dtype == bool
    assert result[EXIT_SIGNAL_COLUMN].dtype == bool
    assert ENTRY_SIGNAL_COLUMN not in candles
    assert EXIT_SIGNAL_COLUMN not in candles


def test_random_forest_direction_model_emits_standard_signal_columns() -> None:
    """Check that the random forest model can run a small walk-forward pass."""

    candles = _sample_candles(180)
    model = RandomForestDirectionModel(
        horizon=2,
        lookback_window=80,
        min_train_size=50,
        retrain_interval=20,
        n_estimators=5,
        max_depth=3,
        min_samples_leaf=2,
    )

    result = model.predict(candles)

    assert ENTRY_SIGNAL_COLUMN in result
    assert EXIT_SIGNAL_COLUMN in result
    assert "random_forest_probability" in result


def test_hist_gradient_boosting_direction_model_emits_standard_columns() -> None:
    """Check that the gradient boosting model can run a small walk-forward pass."""

    candles = _sample_candles(180)
    model = HistGradientBoostingDirectionModel(
        horizon=2,
        lookback_window=80,
        min_train_size=50,
        retrain_interval=20,
        max_iter=5,
        max_leaf_nodes=5,
    )

    result = model.predict(candles)

    assert ENTRY_SIGNAL_COLUMN in result
    assert EXIT_SIGNAL_COLUMN in result
    assert "hist_gradient_boosting_probability" in result


def test_invalid_sma_configuration_raises_error() -> None:
    """Check that invalid model parameters fail early."""

    with pytest.raises(ValueError):
        SmaMomentumModel(fast_window=10, slow_window=5)


def test_warmup_momentum_hold_model_stays_flat_without_momentum() -> None:
    """Check that warmup momentum model does not enter weak initial trends."""

    candles = _sample_candles(80)
    candles["close"] = 100.0
    model = WarmupMomentumHoldModel(lookback_candles=12, minimum_return=0.001)

    result = model.predict(candles)

    assert not result[ENTRY_SIGNAL_COLUMN].any()


def _sample_candles(rows: int) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV candles for tests.

    :param rows: number of rows to generate
    :return: OHLCV dataframe with a datetime index
    """

    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    trend = np.linspace(100.0, 120.0, rows)
    cycle = np.sin(np.arange(rows) / 5.0) * 2.0
    close = pd.Series(trend + cycle, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    volume = pd.Series(1000.0 + np.cos(np.arange(rows) / 7.0) * 100.0, index=index)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
