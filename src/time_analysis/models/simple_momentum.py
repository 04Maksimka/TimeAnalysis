"""Simple baseline models for turning candle data into trading signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAST_SMA_COLUMN = "sma_fast"
SLOW_SMA_COLUMN = "sma_slow"
SCORE_COLUMN = "sma_momentum_score"
ENTRY_SIGNAL_COLUMN = "long_entry_signal"
EXIT_SIGNAL_COLUMN = "long_exit_signal"


@dataclass(frozen=True, slots=True)
class SmaMomentumModel:
    """Generate long-only entry and exit signals from two moving averages.

    The model is deliberately independent from Freqtrade. It accepts a pandas
    dataframe with a ``close`` column and returns a copy with indicator and
    signal columns that any runtime adapter can consume.
    """

    fast_window: int = 12
    slow_window: int = 26

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            msg = "fast_window must be positive"
            raise ValueError(msg)
        if self.slow_window <= 0:
            msg = "slow_window must be positive"
            raise ValueError(msg)
        if self.fast_window >= self.slow_window:
            msg = "fast_window must be smaller than slow_window"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Number of candles needed before the model can emit stable signals."""

        return self.slow_window

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return indicators and entry/exit signal columns for candle data."""

        if "close" not in candles.columns:
            msg = "candles must contain a 'close' column"
            raise ValueError(msg)

        result = candles.copy()
        close = result["close"].astype("float64")

        fast_sma = close.rolling(
            window=self.fast_window,
            min_periods=self.fast_window,
        ).mean()
        slow_sma = close.rolling(
            window=self.slow_window,
            min_periods=self.slow_window,
        ).mean()

        fast_was_below_or_equal = fast_sma.shift(1) <= slow_sma.shift(1)
        fast_was_above_or_equal = fast_sma.shift(1) >= slow_sma.shift(1)

        result[FAST_SMA_COLUMN] = fast_sma
        result[SLOW_SMA_COLUMN] = slow_sma
        result[SCORE_COLUMN] = ((fast_sma / slow_sma) - 1.0).fillna(0.0)
        result[ENTRY_SIGNAL_COLUMN] = (
            (fast_sma > slow_sma) & fast_was_below_or_equal
        ).fillna(False)
        result[EXIT_SIGNAL_COLUMN] = (
            (fast_sma < slow_sma) & fast_was_above_or_equal
        ).fillna(False)

        return result
