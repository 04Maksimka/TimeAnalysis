"""Reusable technical features for trading signal models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index.

    :param close: close price series
    :param window: rolling smoothing window
    :return: RSI values in the ``0`` to ``100`` range
    """

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.fillna(50.0)


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calculate Average True Range.

    :param high: high price series
    :param low: low price series
    :param close: close price series
    :param window: smoothing window
    :return: average true range series
    """

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score values.

    :param series: input series
    :param window: rolling window length
    :return: z-score relative to the rolling mean and standard deviation
    """

    mean = series.rolling(window=window, min_periods=window).mean()
    standard_deviation = series.rolling(window=window, min_periods=window).std(ddof=0)
    return ((series - mean) / standard_deviation.replace(0.0, np.nan)).fillna(0.0)


def safe_percentage_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage change with missing and infinite values cleaned.

    :param series: input numeric series
    :param periods: lag length passed to ``pct_change``
    :return: finite percentage-change series with missing values filled by zero
    """

    changed = series.pct_change(periods=periods)
    return changed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
