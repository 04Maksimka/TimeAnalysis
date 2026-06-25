from __future__ import annotations

# ruff: noqa: I001

from freqtrade.strategy import IStrategy
from pandas import DataFrame

try:
    from .visualization_pipeline import apply_indicators, build_plot_config, load_spec_from_env
except ImportError:  # Freqtrade loads strategy files from strategy_path as top-level modules.
    from visualization_pipeline import apply_indicators, build_plot_config, load_spec_from_env


class ConfigurableVisualStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 100

    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    plot_config = {"main_plot": {}, "subplots": {}}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._visualization_spec = load_spec_from_env(required=False)
        if self._visualization_spec:
            self.plot_config = build_plot_config(self._visualization_spec)
            self.timeframe = self._visualization_spec.get("timeframe", self.timeframe)
            self.startup_candle_count = int(
                self._visualization_spec.get("startup_candle_count", self.startup_candle_count)
            )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self._visualization_spec:
            return dataframe
        return apply_indicators(dataframe, self._visualization_spec)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
