"""Simple baseline models for turning candle data into trading signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from time_analysis.models.base import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    SHORT_ENTRY_SIGNAL_COLUMN,
    SHORT_EXIT_SIGNAL_COLUMN,
    SignalModel,
    crossed_above,
    crossed_below,
    require_columns,
)

FAST_SMA_COLUMN = "sma_fast"
SLOW_SMA_COLUMN = "sma_slow"
SCORE_COLUMN = "sma_momentum_score"


@dataclass(frozen=True, slots=True)
class SmaMomentumModel(SignalModel):
    """Generate long-only entry and exit signals from two moving averages.

    The model is deliberately independent from Freqtrade. It accepts a pandas
    dataframe with a ``close`` column and returns a copy with indicator and
    signal columns that any runtime adapter can consume.

    Attributes:
        fast_window: rolling window for the fast simple moving average
        slow_window: rolling window for the slow simple moving average
    """

    fast_window: int = 12
    slow_window: int = 26

    def __post_init__(self) -> None:
        """Validate moving-average window configuration after initialization."""

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
        """Return the number of candles needed for stable model signals.

        :return: slow moving-average window length
        """

        return self.slow_window

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return indicators and entry/exit signal columns for candle data.

        :param candles: OHLCV dataframe with at least a ``close`` column
        :return: dataframe copy with indicator, score, entry, and exit columns
        """

        require_columns(candles, ["close"])
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

        result[FAST_SMA_COLUMN] = fast_sma
        result[SLOW_SMA_COLUMN] = slow_sma
        result[SCORE_COLUMN] = ((fast_sma / slow_sma) - 1.0).fillna(0.0)
        result[ENTRY_SIGNAL_COLUMN] = crossed_above(fast_sma, slow_sma)
        result[EXIT_SIGNAL_COLUMN] = crossed_below(fast_sma, slow_sma)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = result[EXIT_SIGNAL_COLUMN]
        result[SHORT_EXIT_SIGNAL_COLUMN] = result[ENTRY_SIGNAL_COLUMN]

        return result
