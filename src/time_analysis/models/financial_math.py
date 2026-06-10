"""Rule-based trading models inspired by common financial mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from time_analysis.models.base import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    OHLCV_COLUMNS,
    SHORT_ENTRY_SIGNAL_COLUMN,
    SHORT_EXIT_SIGNAL_COLUMN,
    SignalModel,
    crossed_above,
    crossed_below,
    require_columns,
)
from time_analysis.models.features import average_true_range, relative_strength_index


@dataclass(frozen=True, slots=True)
class BuyAndHoldBenchmarkModel(SignalModel):
    """Benchmark model that enters once and holds the asset.

    Attributes:
        entry_candle: zero-based candle number used for the one-time entry signal
    """

    entry_candle: int = 1

    def __post_init__(self) -> None:
        """Validate buy-and-hold benchmark configuration."""

        if self.entry_candle < 0:
            msg = "entry_candle must be non-negative"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required before the one-time entry.

        :return: configured entry candle index
        """

        return self.entry_candle

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return a one-time entry signal and no exit signal.

        :param candles: OHLCV dataframe with a ``close`` column
        :return: dataframe copy with benchmark signal columns
        """

        require_columns(candles, ["close"])
        result = candles.copy()
        result[ENTRY_SIGNAL_COLUMN] = False
        result[EXIT_SIGNAL_COLUMN] = False
        result[SHORT_ENTRY_SIGNAL_COLUMN] = False
        result[SHORT_EXIT_SIGNAL_COLUMN] = False
        if len(result) > self.entry_candle:
            result.iloc[
                self.entry_candle,
                result.columns.get_loc(ENTRY_SIGNAL_COLUMN),
            ] = True
        return result


@dataclass(frozen=True, slots=True)
class WarmupMomentumHoldModel(SignalModel):
    """Absolute-momentum model that holds only after a positive warmup trend.

    The model waits for a fixed warmup period, measures the return from the
    first candle to the warmup candle, and enters once only when that return is
    greater than a configured threshold. If the warmup trend is weak, the model
    stays in cash.

    Attributes:
        lookback_candles: candles used for the initial momentum check
        minimum_return: minimum warmup return required for a one-time entry
        require_above_average: require warmup close to be above its warmup average
    """

    lookback_candles: int = 72
    minimum_return: float = 0.0
    require_above_average: bool = False

    def __post_init__(self) -> None:
        """Validate warmup momentum configuration."""

        if self.lookback_candles <= 0:
            msg = "lookback_candles must be positive"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required by the warmup trend check.

        :return: configured warmup lookback length
        """

        return self.lookback_candles

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return a one-time entry signal after a positive warmup trend.

        :param candles: OHLCV dataframe with a ``close`` column
        :return: dataframe copy with warmup momentum features and signal columns
        """

        require_columns(candles, ["close"])
        result = candles.copy()
        close = result["close"].astype("float64")
        result["warmup_momentum_return"] = 0.0
        result["warmup_momentum_average"] = close.expanding().mean()
        result[ENTRY_SIGNAL_COLUMN] = False
        result[EXIT_SIGNAL_COLUMN] = False
        result[SHORT_ENTRY_SIGNAL_COLUMN] = False
        result[SHORT_EXIT_SIGNAL_COLUMN] = False

        if len(result) <= self.lookback_candles:
            return result

        warmup_return = (close.iloc[self.lookback_candles] / close.iloc[0]) - 1.0
        warmup_average = close.iloc[: self.lookback_candles + 1].mean()
        result["warmup_momentum_return"] = warmup_return
        average_filter_passed = (
            not self.require_above_average
            or close.iloc[self.lookback_candles] > warmup_average
        )
        if warmup_return > self.minimum_return and average_filter_passed:
            result.iloc[
                self.lookback_candles,
                result.columns.get_loc(ENTRY_SIGNAL_COLUMN),
            ] = True
        return result


@dataclass(frozen=True, slots=True)
class EmaTrendModel(SignalModel):
    """Trend-following model based on fast and slow exponential averages.

    Attributes:
        fast_window: span for the fast exponential moving average
        slow_window: span for the slow exponential moving average
    """

    fast_window: int = 12
    slow_window: int = 48

    def __post_init__(self) -> None:
        """Validate EMA window configuration after initialization."""

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
        """Return the candle count required by the slow EMA.

        :return: slow EMA window length
        """

        return self.slow_window

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return EMA trend features and long-only signals.

        :param candles: OHLCV dataframe with a ``close`` column
        :return: dataframe copy with EMA features and standard signal columns
        """

        require_columns(candles, ["close"])
        result = candles.copy()
        close = result["close"].astype("float64")
        fast_ema = close.ewm(span=self.fast_window, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_window, adjust=False).mean()

        result["ema_fast"] = fast_ema
        result["ema_slow"] = slow_ema
        result["ema_trend_score"] = ((fast_ema / slow_ema) - 1.0).fillna(0.0)
        result[ENTRY_SIGNAL_COLUMN] = crossed_above(fast_ema, slow_ema)
        result[EXIT_SIGNAL_COLUMN] = crossed_below(fast_ema, slow_ema)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = result[EXIT_SIGNAL_COLUMN]
        result[SHORT_EXIT_SIGNAL_COLUMN] = result[ENTRY_SIGNAL_COLUMN]
        return result


@dataclass(frozen=True, slots=True)
class DonchianBreakoutModel(SignalModel):
    """Trend-following breakout model based on Donchian channels.

    Attributes:
        entry_window: lookback window for the prior highest high breakout
        exit_window: lookback window for the prior lowest low exit
    """

    entry_window: int = 55
    exit_window: int = 20

    def __post_init__(self) -> None:
        """Validate Donchian channel window configuration."""

        if self.entry_window <= 1:
            msg = "entry_window must be greater than 1"
            raise ValueError(msg)
        if self.exit_window <= 1:
            msg = "exit_window must be greater than 1"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required by the longest channel.

        :return: maximum of entry and exit windows
        """

        return max(self.entry_window, self.exit_window)

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return Donchian channel features and long-only signals.

        :param candles: OHLCV dataframe with ``high``, ``low``, and ``close``
        :return: dataframe copy with Donchian features and standard signal columns
        """

        require_columns(candles, ["high", "low", "close"])
        result = candles.copy()
        high = result["high"].astype("float64")
        low = result["low"].astype("float64")
        close = result["close"].astype("float64")

        upper = (
            high.shift(1)
            .rolling(
                window=self.entry_window,
                min_periods=self.entry_window,
            )
            .max()
        )
        lower = (
            low.shift(1)
            .rolling(
                window=self.exit_window,
                min_periods=self.exit_window,
            )
            .min()
        )
        middle = (upper + lower) / 2.0

        result["donchian_upper"] = upper
        result["donchian_lower"] = lower
        result["donchian_middle"] = middle
        result["donchian_width"] = ((upper - lower) / close).fillna(0.0)
        result[ENTRY_SIGNAL_COLUMN] = (close > upper).fillna(False)
        result[EXIT_SIGNAL_COLUMN] = (close < lower).fillna(False)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = result[EXIT_SIGNAL_COLUMN]
        result[SHORT_EXIT_SIGNAL_COLUMN] = result[ENTRY_SIGNAL_COLUMN]
        return result


@dataclass(frozen=True, slots=True)
class BollingerRsiMeanReversionModel(SignalModel):
    """Mean-reversion model based on Bollinger Bands and RSI.

    Attributes:
        window: rolling window used for the Bollinger middle band
        standard_deviations: band width multiplier
        rsi_window: RSI smoothing window
        oversold_rsi: maximum RSI value allowed for long entry
        overbought_rsi: minimum RSI value allowed for short entry
        exit_rsi: RSI value that can trigger an exit
    """

    window: int = 20
    standard_deviations: float = 2.0
    rsi_window: int = 14
    oversold_rsi: float = 32.0
    overbought_rsi: float = 68.0
    exit_rsi: float = 55.0

    def __post_init__(self) -> None:
        """Validate Bollinger and RSI configuration."""

        if self.window <= 1:
            msg = "window must be greater than 1"
            raise ValueError(msg)
        if self.standard_deviations <= 0:
            msg = "standard_deviations must be positive"
            raise ValueError(msg)
        if self.rsi_window <= 1:
            msg = "rsi_window must be greater than 1"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required by Bollinger Bands and RSI.

        :return: maximum of Bollinger and RSI windows
        """

        return max(self.window, self.rsi_window)

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return Bollinger/RSI features and long-only signals.

        :param candles: OHLCV dataframe with a ``close`` column
        :return: dataframe copy with Bollinger, RSI, and standard signal columns
        """

        require_columns(candles, ["close"])
        result = candles.copy()
        close = result["close"].astype("float64")

        middle = close.rolling(window=self.window, min_periods=self.window).mean()
        sigma = close.rolling(window=self.window, min_periods=self.window).std(ddof=0)
        upper = middle + self.standard_deviations * sigma
        lower = middle - self.standard_deviations * sigma
        rsi = relative_strength_index(close, self.rsi_window)

        result["bb_middle"] = middle
        result["bb_upper"] = upper
        result["bb_lower"] = lower
        result["bb_percent_b"] = ((close - lower) / (upper - lower)).fillna(0.5)
        result["rsi"] = rsi
        result[ENTRY_SIGNAL_COLUMN] = (
            (close <= lower) & (rsi <= self.oversold_rsi)
        ).fillna(False)
        result[EXIT_SIGNAL_COLUMN] = (
            (close >= middle) | (rsi >= self.exit_rsi)
        ).fillna(False)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = (
            (close >= upper) & (rsi >= self.overbought_rsi)
        ).fillna(False)
        result[SHORT_EXIT_SIGNAL_COLUMN] = (
            (close <= middle) | (rsi <= 100.0 - self.exit_rsi)
        ).fillna(False)
        return result


@dataclass(frozen=True, slots=True)
class MacdRsiTrendModel(SignalModel):
    """Momentum model that combines MACD histogram crosses with RSI filters.

    Attributes:
        fast_window: fast EMA span for MACD
        slow_window: slow EMA span for MACD
        signal_window: EMA span for the MACD signal line
        rsi_window: RSI smoothing window
        min_entry_rsi: minimum RSI value allowed for entry
        max_entry_rsi: maximum RSI value allowed for entry
        exit_rsi: RSI value below which an exit is emitted
    """

    fast_window: int = 12
    slow_window: int = 26
    signal_window: int = 9
    rsi_window: int = 14
    min_entry_rsi: float = 50.0
    max_entry_rsi: float = 75.0
    exit_rsi: float = 45.0

    def __post_init__(self) -> None:
        """Validate MACD and RSI configuration."""

        if self.fast_window <= 0:
            msg = "fast_window must be positive"
            raise ValueError(msg)
        if self.slow_window <= 0:
            msg = "slow_window must be positive"
            raise ValueError(msg)
        if self.fast_window >= self.slow_window:
            msg = "fast_window must be smaller than slow_window"
            raise ValueError(msg)
        if self.signal_window <= 0:
            msg = "signal_window must be positive"
            raise ValueError(msg)
        if self.rsi_window <= 1:
            msg = "rsi_window must be greater than 1"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required by MACD and RSI.

        :return: maximum indicator window length
        """

        return max(self.slow_window + self.signal_window, self.rsi_window)

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return MACD/RSI features and long-only signals.

        :param candles: OHLCV dataframe with a ``close`` column
        :return: dataframe copy with MACD, RSI, and standard signal columns
        """

        require_columns(candles, ["close"])
        result = candles.copy()
        close = result["close"].astype("float64")

        fast_ema = close.ewm(span=self.fast_window, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_window, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()
        histogram = macd - signal
        rsi = relative_strength_index(close, self.rsi_window)

        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_histogram"] = histogram
        result["rsi"] = rsi
        result[ENTRY_SIGNAL_COLUMN] = (
            crossed_above(histogram, 0.0)
            & (rsi >= self.min_entry_rsi)
            & (rsi <= self.max_entry_rsi)
        )
        result[EXIT_SIGNAL_COLUMN] = (
            crossed_below(histogram, 0.0) | (rsi <= self.exit_rsi)
        ).fillna(False)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = (
            crossed_below(histogram, 0.0)
            & (rsi <= self.min_entry_rsi)
            & (rsi >= 100.0 - self.max_entry_rsi)
        )
        result[SHORT_EXIT_SIGNAL_COLUMN] = (
            crossed_above(histogram, 0.0) | (rsi >= 100.0 - self.exit_rsi)
        ).fillna(False)
        return result


@dataclass(frozen=True, slots=True)
class AtrVolatilityBreakoutModel(SignalModel):
    """Volatility breakout model based on ATR and a long EMA regime filter.

    Attributes:
        atr_window: ATR smoothing window
        ema_window: EMA window used as a trend regime filter
        breakout_multiplier: ATR multiplier above the prior close for entry
        exit_multiplier: ATR multiplier below the prior close for exit
    """

    atr_window: int = 14
    ema_window: int = 96
    breakout_multiplier: float = 1.2
    exit_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate ATR volatility breakout configuration."""

        if self.atr_window <= 1:
            msg = "atr_window must be greater than 1"
            raise ValueError(msg)
        if self.ema_window <= 1:
            msg = "ema_window must be greater than 1"
            raise ValueError(msg)
        if self.breakout_multiplier <= 0:
            msg = "breakout_multiplier must be positive"
            raise ValueError(msg)
        if self.exit_multiplier <= 0:
            msg = "exit_multiplier must be positive"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required by ATR and EMA filters.

        :return: maximum of ATR and EMA windows
        """

        return max(self.atr_window, self.ema_window)

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return ATR breakout features and long-only signals.

        :param candles: OHLCV dataframe with standard OHLCV columns
        :return: dataframe copy with ATR, regime, and standard signal columns
        """

        require_columns(candles, OHLCV_COLUMNS)
        result = candles.copy()
        close = result["close"].astype("float64")
        high = result["high"].astype("float64")
        low = result["low"].astype("float64")
        atr = average_true_range(high, low, close, self.atr_window)
        trend_ema = close.ewm(span=self.ema_window, adjust=False).mean()
        prior_close = close.shift(1)
        entry_level = prior_close + self.breakout_multiplier * atr.shift(1)
        exit_level = prior_close - self.exit_multiplier * atr.shift(1)

        result["atr"] = atr
        result["atr_entry_level"] = entry_level
        result["atr_exit_level"] = exit_level
        result["atr_trend_ema"] = trend_ema
        result[ENTRY_SIGNAL_COLUMN] = (
            (close > entry_level) & (close > trend_ema)
        ).fillna(False)
        result[EXIT_SIGNAL_COLUMN] = (
            (close < exit_level) | (close < trend_ema)
        ).fillna(False)
        result[SHORT_ENTRY_SIGNAL_COLUMN] = (
            (close < exit_level) & (close < trend_ema)
        ).fillna(False)
        result[SHORT_EXIT_SIGNAL_COLUMN] = (
            (close > entry_level) | (close > trend_ema)
        ).fillna(False)
        return result
