"""Plotly visualizations for Freqtrade backtest reports."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from time_analysis.backtesting.freqtrade_report import FreqtradeBacktestReport


def build_backtest_dashboard(report: FreqtradeBacktestReport) -> dict[str, Any]:
    """Build report tables and core dashboard figures for a notebook.

    :param report: parsed Freqtrade backtest report
    :return: dictionary with metrics, tables, and Plotly figures
    """

    return {
        "metrics": report.metrics(),
        "trades": report.trades_frame(),
        "pairs": report.pair_summary_frame(),
        "exit_reasons": report.exit_reason_summary_frame(),
        "daily_profit": report.daily_profit_frame(),
        "equity_curve": report.equity_curve_frame(),
        "figures": {
            "equity": plot_equity_curve(report),
            "drawdown": plot_drawdown(report),
            "daily_profit": plot_daily_profit(report),
            "pair_summary": plot_pair_summary(report),
        },
    }


def plot_equity_curve(report: FreqtradeBacktestReport) -> go.Figure:
    """Plot trade-by-trade account equity.

    :param report: parsed Freqtrade backtest report
    :return: Plotly figure with account equity after closed trades
    """

    frame = report.equity_curve_frame()
    fig = go.Figure()
    if frame.empty:
        return _empty_figure("Equity curve", "No closed trades found")

    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["equity"],
            mode="lines+markers",
            name="Equity",
        )
    )
    fig.update_layout(
        title="Equity curve",
        xaxis_title="Time",
        yaxis_title="Account value",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_drawdown(report: FreqtradeBacktestReport) -> go.Figure:
    """Plot drawdown percentage after each closed trade.

    :param report: parsed Freqtrade backtest report
    :return: Plotly figure with drawdown percentage over time
    """

    frame = report.equity_curve_frame()
    fig = go.Figure()
    if frame.empty:
        return _empty_figure("Drawdown", "No closed trades found")

    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["drawdown_ratio"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown, %",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_daily_profit(report: FreqtradeBacktestReport) -> go.Figure:
    """Plot daily absolute profit and cumulative daily profit.

    :param report: parsed Freqtrade backtest report
    :return: Plotly figure with daily and cumulative absolute profit
    """

    frame = report.daily_profit_frame()
    fig = go.Figure()
    if frame.empty:
        return _empty_figure("Daily profit", "No daily profit rows found")

    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["profit_abs"],
            name="Daily profit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["cumulative_profit_abs"],
            mode="lines+markers",
            name="Cumulative profit",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Daily profit",
        xaxis_title="Date",
        yaxis_title="Daily profit",
        yaxis2={
            "title": "Cumulative profit",
            "overlaying": "y",
            "side": "right",
        },
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_pair_summary(report: FreqtradeBacktestReport) -> go.Figure:
    """Plot per-pair profit from the Freqtrade summary table.

    :param report: parsed Freqtrade backtest report
    :return: Plotly figure with horizontal per-pair profit bars
    """

    frame = report.pair_summary_frame()
    fig = go.Figure()
    if frame.empty:
        return _empty_figure("Pair summary", "No pair summary rows found")

    pair_column = "pair" if "pair" in frame else "key"
    profit_column = _first_existing_column(
        frame,
        ["profit_total_abs", "profit_abs", "profit_total", "profit_mean"],
    )
    if profit_column is None:
        return _empty_figure("Pair summary", "No profit column found")

    plot_frame = frame.sort_values(profit_column, ascending=True)
    fig.add_trace(
        go.Bar(
            x=plot_frame[profit_column],
            y=plot_frame[pair_column],
            orientation="h",
            name="Profit",
        )
    )
    fig.update_layout(
        title="Pair summary",
        xaxis_title=profit_column,
        yaxis_title="Pair",
        template="plotly_white",
    )
    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    """Build a placeholder figure for empty report sections.

    :param title: figure title
    :param message: centered annotation text
    :return: Plotly figure with one annotation
    """

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def _first_existing_column(frame: pd.DataFrame, columns: list[str]) -> str | None:
    """Return the first candidate column present in a dataframe.

    :param frame: dataframe to inspect
    :param columns: ordered candidate column names
    :return: first existing column name, or ``None`` when none are present
    """

    return next((column for column in columns if column in frame), None)
