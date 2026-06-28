from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy


class VisualOnlyStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    startup_candle_count = 100

    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    plot_config = {
        "main_plot": {
            "sma_20": {"color": "#f59e0b"},
            "sma_50": {"color": "#2563eb"},
            "ema_20": {"color": "#10b981"},
            "ema_50": {"color": "#ef4444"},
            "bb_lowerband": {"color": "#94a3b8"},
            "bb_upperband": {
                "color": "#94a3b8",
                "fill_to": "bb_lowerband",
                "fill_color": "rgba(148, 163, 184, 0.12)",
            },
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "#7c3aed"},
            },
            "MACD": {
                "macd": {"color": "#2563eb"},
                "macdsignal": {"color": "#f97316"},
                "macdhist": {"type": "bar", "plotly": {"opacity": 0.6}},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma_20"] = ta.SMA(dataframe, timeperiod=20)
        dataframe["sma_50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_lowerband"] = bollinger["lowerband"]
        dataframe["bb_upperband"] = bollinger["upperband"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
