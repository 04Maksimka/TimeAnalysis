from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from time_analysis.backtesting.freqtrade_report import FreqtradeBacktestReport
from time_analysis.backtesting.visualization import (
    build_backtest_dashboard,
    plot_daily_profit,
    plot_drawdown,
    plot_equity_curve,
    plot_pair_summary,
)


def test_build_backtest_dashboard_returns_tables_and_figures() -> None:
    """Check that dashboard contains report tables and all core figures."""

    report = _sample_report()

    dashboard = build_backtest_dashboard(report)

    assert dashboard["metrics"]["total_trades"] == 2
    assert len(dashboard["trades"]) == 2
    assert len(dashboard["pairs"]) == 2
    assert set(dashboard["figures"]) == {
        "equity",
        "drawdown",
        "daily_profit",
        "pair_summary",
    }


def test_core_plots_return_plotly_figures() -> None:
    """Check that all core plot helpers return Plotly figures."""

    report = _sample_report()

    assert isinstance(plot_equity_curve(report), go.Figure)
    assert isinstance(plot_drawdown(report), go.Figure)
    assert isinstance(plot_daily_profit(report), go.Figure)
    assert isinstance(plot_pair_summary(report), go.Figure)


def test_empty_report_plots_are_still_figures() -> None:
    """Check that empty reports render placeholder Plotly figures."""

    report = FreqtradeBacktestReport(
        source_path=Path("empty.json"),
        strategy_name="EmptyStrategy",
        payload={},
    )

    assert isinstance(plot_equity_curve(report), go.Figure)
    assert isinstance(plot_drawdown(report), go.Figure)
    assert isinstance(plot_daily_profit(report), go.Figure)
    assert isinstance(plot_pair_summary(report), go.Figure)


def _sample_report() -> FreqtradeBacktestReport:
    """Build a small deterministic Freqtrade report fixture.

    :return: report with two trades, two pairs, and daily profit rows
    """

    return FreqtradeBacktestReport(
        source_path=Path("backtest-result.json"),
        strategy_name="DemoStrategy",
        payload={
            "dry_run_wallet": 1000.0,
            "total_trades": 2,
            "wins": 1,
            "losses": 1,
            "winrate": 0.5,
            "profit_total_abs": 3.0,
            "trades": [
                {
                    "pair": "BTC/USDT",
                    "open_date": "2025-01-01 00:00:00+00:00",
                    "close_date": "2025-01-01 01:00:00+00:00",
                    "profit_abs": 5.0,
                    "profit_ratio": 0.01,
                },
                {
                    "pair": "ETH/USDT",
                    "open_date": "2025-01-02 00:00:00+00:00",
                    "close_date": "2025-01-02 01:00:00+00:00",
                    "profit_abs": -2.0,
                    "profit_ratio": -0.004,
                },
            ],
            "results_per_pair": [
                {
                    "key": "BTC/USDT",
                    "trades": 1,
                    "profit_total_abs": 5.0,
                },
                {
                    "key": "ETH/USDT",
                    "trades": 1,
                    "profit_total_abs": -2.0,
                },
            ],
            "daily_profit": [
                ["2025-01-01", 5.0],
                ["2025-01-02", -2.0],
            ],
        },
    )
