"""Backtesting utilities for runtime-specific result formats."""

from time_analysis.backtesting.freqtrade_pipeline import (
    BacktestExperimentConfig,
    FreqtradePipelineError,
    build_backtesting_command,
    build_download_data_command,
    ensure_freqtrade_config,
    run_checked_freqtrade_command,
    run_freqtrade_backtest,
    run_freqtrade_command,
)
from time_analysis.backtesting.freqtrade_report import (
    FreqtradeBacktestReport,
    find_latest_backtest_result,
    load_backtest_payload,
    load_latest_backtest_report,
)
from time_analysis.backtesting.signal_backtester import (
    SignalBacktestConfig,
    SignalBacktestResult,
    compare_signal_models,
    compare_signal_models_across_pairs,
    evaluate_signal_model,
    find_freqtrade_ohlcv_file,
    load_freqtrade_ohlcv,
)
from time_analysis.backtesting.visualization import (
    build_backtest_dashboard,
    plot_daily_profit,
    plot_drawdown,
    plot_equity_curve,
    plot_pair_summary,
)

__all__ = [
    "BacktestExperimentConfig",
    "FreqtradeBacktestReport",
    "FreqtradePipelineError",
    "SignalBacktestConfig",
    "SignalBacktestResult",
    "build_backtest_dashboard",
    "build_backtesting_command",
    "build_download_data_command",
    "compare_signal_models",
    "compare_signal_models_across_pairs",
    "evaluate_signal_model",
    "find_latest_backtest_result",
    "find_freqtrade_ohlcv_file",
    "ensure_freqtrade_config",
    "load_backtest_payload",
    "load_freqtrade_ohlcv",
    "load_latest_backtest_report",
    "plot_daily_profit",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_pair_summary",
    "run_checked_freqtrade_command",
    "run_freqtrade_backtest",
    "run_freqtrade_command",
]
