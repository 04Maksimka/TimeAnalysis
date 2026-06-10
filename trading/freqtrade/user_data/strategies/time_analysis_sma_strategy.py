from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame

from time_analysis.models.simple_momentum import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    SmaMomentumModel,
)


class TimeAnalysisSmaStrategy(IStrategy):
    """Freqtrade adapter for the baseline TimeAnalysis SMA model.

    Attributes:
        timeframe: Freqtrade candle timeframe used by the strategy
        can_short: whether the strategy can open short positions
        process_only_new_candles: process only completed new candles
        model: exchange-independent pandas signal model
        startup_candle_count: candles required before stable signals are emitted
    """

    timeframe = "5m"
    can_short = False
    process_only_new_candles = True

    model = SmaMomentumModel(fast_window=12, slow_window=26)
    startup_candle_count = model.startup_candle_count

    minimal_roi = {
        "0": 0.03,
        "120": 0.01,
        "360": 0.0,
    }
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC",
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate model indicators and raw signal columns.

        :param dataframe: Freqtrade OHLCV dataframe for the current pair
        :param metadata: Freqtrade metadata with pair and runtime context
        :return: dataframe with model indicator and signal columns
        """

        return self.model.predict(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Map the model entry signal to Freqtrade's entry column.

        :param dataframe: dataframe returned by ``populate_indicators``
        :param metadata: Freqtrade metadata with pair and runtime context
        :return: dataframe with ``enter_long`` values populated
        """

        dataframe["enter_long"] = 0
        dataframe.loc[dataframe[ENTRY_SIGNAL_COLUMN], "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Map the model exit signal to Freqtrade's exit column.

        :param dataframe: dataframe returned by ``populate_indicators``
        :param metadata: Freqtrade metadata with pair and runtime context
        :return: dataframe with ``exit_long`` values populated
        """

        dataframe["exit_long"] = 0
        dataframe.loc[dataframe[EXIT_SIGNAL_COLUMN], "exit_long"] = 1
        return dataframe
