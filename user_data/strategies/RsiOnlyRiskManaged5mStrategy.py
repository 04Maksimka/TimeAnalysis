import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from freqtrade.vendor.qtpylib import indicators as qtpylib
from pandas import DataFrame


class RsiOnlyRiskManaged5mStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 100

    minimal_roi = {"0": 0.1}
    stoploss = -0.1
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_rsi_period = IntParameter(7, 30, default=14, space="buy")
    buy_rsi_lower = IntParameter(10, 45, default=30, space="buy")
    sell_rsi_upper = IntParameter(55, 90, default=70, space="sell")

    protection_cooldown_stop_duration = IntParameter(
        3, 36, default=12, space="protection"
    )
    protection_max_drawdown_stop_duration = IntParameter(
        24, 288, default=72, space="protection"
    )
    protection_max_allowed_drawdown = DecimalParameter(
        0.05, 0.25, decimals=3, default=0.15, space="protection"
    )
    protection_low_profit_stop_duration = IntParameter(
        24, 288, default=72, space="protection"
    )
    protection_low_profit_required_profit = DecimalParameter(
        -0.02, 0.03, decimals=3, default=0.0, space="protection"
    )

    plot_config = {
        "subplots": {
            "RSI": {
                "rsi": {"color": "#7c3aed"},
                "rsi_lower": {"color": "#22c55e"},
                "rsi_upper": {"color": "#ef4444"},
            },
        },
    }

    @property
    def protections(self) -> list[dict]:
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.protection_cooldown_stop_duration.value,
            },
            {
                "method": "MaxDrawdown",
                "calculation_mode": "equity",
                "lookback_period_candles": 288,
                "trade_limit": 20,
                "stop_duration_candles": (
                    self.protection_max_drawdown_stop_duration.value
                ),
                "max_allowed_drawdown": self.protection_max_allowed_drawdown.value,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 288,
                "trade_limit": 3,
                "stop_duration_candles": self.protection_low_profit_stop_duration.value,
                "required_profit": self.protection_low_profit_required_profit.value,
            },
        ]

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
