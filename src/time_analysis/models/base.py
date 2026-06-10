"""Shared interface and helpers for exchange-independent signal models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

ENTRY_SIGNAL_COLUMN = "long_entry_signal"
EXIT_SIGNAL_COLUMN = "long_exit_signal"
SHORT_ENTRY_SIGNAL_COLUMN = "short_entry_signal"
SHORT_EXIT_SIGNAL_COLUMN = "short_exit_signal"
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class SignalModel(ABC):
    """Base class for all pandas-based trading signal models.

    A signal model receives historical candles and returns a dataframe copy with
    feature columns plus the standard entry and exit signal columns. Long
    signals are required. Short signals are optional and used only by evaluators
    configured for long/short research.
    """

    @property
    def name(self) -> str:
        """Return the model display name.

        :return: class name used in reports and comparison tables
        """

        return self.__class__.__name__

    @property
    @abstractmethod
    def startup_candle_count(self) -> int:
        """Return the number of candles needed before stable signals exist.

        :return: minimum candle count required by the model
        """

    @abstractmethod
    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return model features and standard signal columns.

        :param candles: OHLCV dataframe sorted from oldest to newest candle
        :return: dataframe copy with standard entry and exit signal columns
        """


def require_columns(candles: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that a candle dataframe contains required columns.

    :param candles: dataframe to validate
    :param columns: required column names
    """

    missing_columns = [column for column in columns if column not in candles]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        msg = f"candles must contain required columns: {joined_columns}"
        raise ValueError(msg)


def crossed_above(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """Return points where one series crosses above another value.

    :param left: left-hand series
    :param right: right-hand series or scalar threshold
    :return: boolean series that is true only on the upward crossing candle
    """

    right_series = _as_aligned_series(right, left.index)
    return ((left > right_series) & (left.shift(1) <= right_series.shift(1))).fillna(
        False
    )


def crossed_below(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """Return points where one series crosses below another value.

    :param left: left-hand series
    :param right: right-hand series or scalar threshold
    :return: boolean series that is true only on the downward crossing candle
    """

    right_series = _as_aligned_series(right, left.index)
    return ((left < right_series) & (left.shift(1) >= right_series.shift(1))).fillna(
        False
    )


def _as_aligned_series(value: pd.Series | float, index: pd.Index) -> pd.Series:
    """Convert a scalar or series to a series aligned to an index.

    :param value: scalar value or existing series
    :param index: target index
    :return: series aligned to the target index
    """

    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)
