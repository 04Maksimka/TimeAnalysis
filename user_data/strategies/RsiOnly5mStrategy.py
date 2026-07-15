import talib.abstract as ta
from freqtrade.strategy import IntParameter, IStrategy
from freqtrade.vendor.qtpylib import indicators as qtpylib
from pandas import DataFrame


class RsiOnly5mStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 100

    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_rsi_period = IntParameter(7, 30, default=14, space="buy")
    buy_rsi_lower = IntParameter(10, 45, default=30, space="buy")
    sell_rsi_upper = IntParameter(55, 90, default=70, space="sell")

    plot_config = {
        "subplots": {
            "RSI": {
                "rsi": {"color": "#7c3aed"},
                "rsi_lower": {"color": "#22c55e"},
                "rsi_upper": {"color": "#ef4444"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for period in self.buy_rsi_period.range:
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        dataframe["rsi"] = dataframe[f"rsi_{self.buy_rsi_period.value}"]
        dataframe["rsi_lower"] = self.buy_rsi_lower.value
        dataframe["rsi_upper"] = self.sell_rsi_upper.value

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi = dataframe[f"rsi_{self.buy_rsi_period.value}"]

        dataframe.loc[
            (
                qtpylib.crossed_below(rsi, self.buy_rsi_lower.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi = dataframe[f"rsi_{self.buy_rsi_period.value}"]

        dataframe.loc[
            (
                qtpylib.crossed_above(rsi, self.sell_rsi_upper.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
