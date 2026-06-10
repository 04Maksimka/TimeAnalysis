"""Vectorized research backtester for pandas signal models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from time_analysis.models.base import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    SHORT_ENTRY_SIGNAL_COLUMN,
    SHORT_EXIT_SIGNAL_COLUMN,
    SignalModel,
    require_columns,
)


@dataclass(frozen=True, slots=True)
class SignalBacktestConfig:
    """Configuration for one vectorized signal-model backtest.

    Attributes:
        initial_balance: starting account value
        fee_rate: proportional cost applied when the position changes
        periods_per_year: annualization factor for Sharpe and annual return
        price_column: candle price column used for returns and fills
        allow_short: use optional short signal columns when the model emits them
    """

    initial_balance: float = 1000.0
    fee_rate: float = 0.001
    periods_per_year: int = 365 * 24 * 12
    price_column: str = "close"
    allow_short: bool = False


@dataclass(frozen=True, slots=True)
class SignalBacktestResult:
    """Result of one vectorized signal-model backtest.

    Attributes:
        model_name: display name of the evaluated model
        signals: model output dataframe with features and signals
        equity_curve: dataframe with returns, position, equity, and drawdown
        trades: dataframe with reconstructed closed trades
        metrics: summary metrics for model comparison
    """

    model_name: str
    signals: pd.DataFrame
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int | str]


def evaluate_signal_model(
    candles: pd.DataFrame,
    model: SignalModel,
    config: SignalBacktestConfig | None = None,
) -> SignalBacktestResult:
    """Evaluate one signal model with a vectorized backtest.

    Signals emitted on candle ``t`` control exposure for the return from
    ``t`` to ``t + 1``. This avoids using the current candle's return before
    the signal could have been known.

    :param candles: OHLCV dataframe sorted from oldest to newest candle
    :param model: signal model implementing the shared ``SignalModel`` interface
    :param config: optional vectorized backtest configuration
    :return: complete result with signals, equity, trades, and metrics
    """

    selected_config = config or SignalBacktestConfig()
    require_columns(candles, [selected_config.price_column])

    signals = model.predict(candles)
    require_columns(signals, [ENTRY_SIGNAL_COLUMN, EXIT_SIGNAL_COLUMN])

    close = signals[selected_config.price_column].astype("float64")
    short_entry_signal = _optional_signal(signals, SHORT_ENTRY_SIGNAL_COLUMN)
    short_exit_signal = _optional_signal(signals, SHORT_EXIT_SIGNAL_COLUMN)
    position = _build_position(
        entry_signal=signals[ENTRY_SIGNAL_COLUMN].astype("bool"),
        exit_signal=signals[EXIT_SIGNAL_COLUMN].astype("bool"),
        short_entry_signal=short_entry_signal,
        short_exit_signal=short_exit_signal,
        allow_short=selected_config.allow_short,
    )
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exposure = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = (exposure * returns) - (turnover * selected_config.fee_rate)
    equity = selected_config.initial_balance * (1.0 + strategy_returns).cumprod()
    equity_peak = equity.cummax()
    drawdown = (equity / equity_peak) - 1.0

    equity_curve = pd.DataFrame(
        {
            "price": close,
            "position": position,
            "return": returns,
            "strategy_return": strategy_returns,
            "equity": equity,
            "equity_peak": equity_peak,
            "drawdown": drawdown,
        },
        index=signals.index,
    )
    trades = _reconstruct_trades(
        close=close,
        position=position,
        fee_rate=selected_config.fee_rate,
    )
    metrics = _calculate_metrics(
        model_name=model.name,
        equity_curve=equity_curve,
        trades=trades,
        initial_balance=selected_config.initial_balance,
        periods_per_year=selected_config.periods_per_year,
    )
    return SignalBacktestResult(
        model_name=model.name,
        signals=signals,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
    )


def compare_signal_models(
    candles: pd.DataFrame,
    models: tuple[SignalModel, ...] | list[SignalModel],
    config: SignalBacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, SignalBacktestResult]]:
    """Evaluate several signal models on the same candle dataframe.

    :param candles: OHLCV dataframe sorted from oldest to newest candle
    :param models: model instances implementing the shared interface
    :param config: optional vectorized backtest configuration
    :return: metrics dataframe and mapping from model name to full result
    """

    results = {
        model.name: evaluate_signal_model(candles, model, config) for model in models
    }
    metrics = pd.DataFrame([result.metrics for result in results.values()])
    if metrics.empty:
        return metrics, results
    metrics = metrics.sort_values(
        ["total_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return metrics, results


def compare_signal_models_across_pairs(
    candles_by_pair: dict[str, pd.DataFrame],
    models: tuple[SignalModel, ...] | list[SignalModel],
    config: SignalBacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, SignalBacktestResult]]]:
    """Evaluate signal models across several pairs and aggregate metrics.

    :param candles_by_pair: mapping from pair name to OHLCV dataframe
    :param models: model instances implementing the shared interface
    :param config: optional vectorized backtest configuration
    :return: pair metrics, aggregate metrics, and full results by pair
    """

    pair_metric_frames: list[pd.DataFrame] = []
    results_by_pair: dict[str, dict[str, SignalBacktestResult]] = {}
    for pair, candles in candles_by_pair.items():
        pair_metrics, pair_results = compare_signal_models(
            candles=candles,
            models=models,
            config=config,
        )
        pair_metrics = pair_metrics.copy()
        pair_metrics.insert(0, "pair", pair)
        pair_metric_frames.append(pair_metrics)
        results_by_pair[pair] = pair_results

    if not pair_metric_frames:
        return pd.DataFrame(), pd.DataFrame(), results_by_pair

    pair_metrics = pd.concat(pair_metric_frames, ignore_index=True)
    aggregate_metrics = _aggregate_pair_metrics(pair_metrics)
    return pair_metrics, aggregate_metrics, results_by_pair


def _aggregate_pair_metrics(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pair model metrics into one comparison table.

    :param pair_metrics: metrics dataframe returned by pair-level comparisons
    :return: one row per model with portfolio-style summary metrics
    """

    grouped = pair_metrics.groupby("model", as_index=False).agg(
        average_return=("total_return", "mean"),
        minimum_return=("total_return", "min"),
        maximum_return=("total_return", "max"),
        positive_pairs=("total_return", lambda values: int((values > 0.0).sum())),
        non_negative_pairs=(
            "total_return",
            lambda values: int((values >= 0.0).sum()),
        ),
        worst_drawdown=("max_drawdown", "min"),
        average_sharpe=("sharpe_ratio", "mean"),
        total_trades=("total_trades", "sum"),
        average_win_rate=("win_rate", "mean"),
        average_profit_factor=("profit_factor", "mean"),
    )
    return grouped.sort_values(
        ["average_return", "minimum_return", "average_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def find_freqtrade_ohlcv_file(
    freqtrade_dir: Path | str,
    pair: str,
    timeframe: str,
) -> Path:
    """Find a downloaded Freqtrade OHLCV file for one pair and timeframe.

    :param freqtrade_dir: local ``trading/freqtrade`` directory
    :param pair: exchange pair such as ``BTC/USDT``
    :param timeframe: candle timeframe such as ``5m``
    :return: path to the matching OHLCV data file
    """

    data_dir = Path(freqtrade_dir) / "user_data" / "data"
    pair_slug = pair.replace("/", "_").replace(":", "_")
    supported_suffixes = {".feather", ".parquet", ".csv", ".json"}
    candidates = [
        path
        for path in data_dir.rglob(f"*{pair_slug}*{timeframe}*")
        if path.is_file() and path.suffix in supported_suffixes
    ]
    if not candidates:
        msg = f"No Freqtrade OHLCV file found for {pair} {timeframe} in {data_dir}"
        raise FileNotFoundError(msg)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_freqtrade_ohlcv(
    freqtrade_dir: Path | str,
    pair: str,
    timeframe: str,
) -> pd.DataFrame:
    """Load downloaded Freqtrade OHLCV candles into a dataframe.

    :param freqtrade_dir: local ``trading/freqtrade`` directory
    :param pair: exchange pair such as ``BTC/USDT``
    :param timeframe: candle timeframe such as ``5m``
    :return: OHLCV dataframe sorted by date
    """

    path = find_freqtrade_ohlcv_file(freqtrade_dir, pair, timeframe)
    if path.suffix == ".feather":
        candles = pd.read_feather(path)
    elif path.suffix == ".parquet":
        candles = pd.read_parquet(path)
    elif path.suffix == ".csv":
        candles = pd.read_csv(path)
    elif path.suffix == ".json":
        candles = pd.read_json(path)
    else:
        msg = f"Unsupported OHLCV file format: {path.suffix}"
        raise ValueError(msg)

    if "date" in candles:
        candles["date"] = pd.to_datetime(candles["date"], utc=True)
        candles = candles.sort_values("date").set_index("date")
    return candles.sort_index()


def _optional_signal(signals: pd.DataFrame, column: str) -> pd.Series:
    """Return an optional boolean signal column or an all-false fallback.

    :param signals: model output dataframe
    :param column: optional signal column name
    :return: boolean signal series aligned to ``signals``
    """

    if column in signals:
        return signals[column].astype("bool")
    return pd.Series(False, index=signals.index, dtype="bool")


def _build_position(
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    short_entry_signal: pd.Series,
    short_exit_signal: pd.Series,
    allow_short: bool,
) -> pd.Series:
    """Build a position series from long and optional short events.

    :param entry_signal: boolean series with entry events
    :param exit_signal: boolean series with exit events
    :param short_entry_signal: boolean series with short entry events
    :param short_exit_signal: boolean series with short exit events
    :param allow_short: whether short signals can open short exposure
    :return: position series with values ``-1.0``, ``0.0``, or ``1.0``
    """

    position = 0.0
    positions: list[float] = []
    for long_entry, long_exit, short_entry, short_exit in zip(
        entry_signal,
        exit_signal,
        short_entry_signal,
        short_exit_signal,
        strict=False,
    ):
        if (
            position == 1.0
            and (long_exit or (allow_short and short_entry))
            or position == -1.0
            and (short_exit or long_entry)
        ):
            position = 0.0

        if position == 0.0 and long_entry:
            position = 1.0
        elif allow_short and position == 0.0 and short_entry:
            position = -1.0

        positions.append(position)
    return pd.Series(positions, index=entry_signal.index, dtype="float64")


def _reconstruct_trades(
    close: pd.Series,
    position: pd.Series,
    fee_rate: float,
) -> pd.DataFrame:
    """Reconstruct closed trades from a position series.

    :param close: price series used for fills
    :param position: long-only position series
    :param fee_rate: proportional cost charged on entry and exit
    :return: dataframe with one row per reconstructed trade
    """

    rows: list[dict[str, object]] = []
    entry_time = None
    entry_price = None
    entry_side = None
    previous_position = 0.0
    for time, current_position in position.items():
        price = float(close.loc[time])
        if previous_position == 0.0 and current_position != 0.0:
            entry_time = time
            entry_price = price
            entry_side = "long" if current_position > 0.0 else "short"
        elif previous_position != 0.0 and current_position == 0.0 and entry_price:
            if previous_position > 0.0:
                gross_return = (price / entry_price) - 1.0
            else:
                gross_return = (entry_price / price) - 1.0
            net_return = gross_return - (2.0 * fee_rate)
            rows.append(
                {
                    "side": entry_side,
                    "entry_time": entry_time,
                    "exit_time": time,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                }
            )
            entry_time = None
            entry_price = None
            entry_side = None
        previous_position = current_position

    if previous_position != 0.0 and entry_price:
        price = float(close.iloc[-1])
        if previous_position > 0.0:
            gross_return = (price / entry_price) - 1.0
        else:
            gross_return = (entry_price / price) - 1.0
        net_return = gross_return - (2.0 * fee_rate)
        rows.append(
            {
                "side": entry_side,
                "entry_time": entry_time,
                "exit_time": close.index[-1],
                "entry_price": entry_price,
                "exit_price": price,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
    return pd.DataFrame(rows)


def _calculate_metrics(
    model_name: str,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_balance: float,
    periods_per_year: int,
) -> dict[str, float | int | str]:
    """Calculate summary metrics for one signal-model backtest.

    :param model_name: display name of the evaluated model
    :param equity_curve: dataframe returned by the vectorized backtest
    :param trades: reconstructed trades dataframe
    :param initial_balance: starting account value
    :param periods_per_year: annualization factor
    :return: dictionary of scalar comparison metrics
    """

    final_equity = float(equity_curve["equity"].iloc[-1])
    total_return = (final_equity / initial_balance) - 1.0
    strategy_returns = equity_curve["strategy_return"]
    mean_return = float(strategy_returns.mean())
    return_std = float(strategy_returns.std(ddof=0))
    sharpe_ratio = 0.0
    if return_std > 0.0:
        sharpe_ratio = (mean_return / return_std) * float(np.sqrt(periods_per_year))
    annual_return = (1.0 + total_return) ** (
        periods_per_year / max(len(equity_curve), 1)
    ) - 1.0
    max_drawdown = float(equity_curve["drawdown"].min())
    exposure = float(equity_curve["position"].abs().mean())

    total_trades = int(len(trades))
    win_rate = 0.0
    profit_factor = 0.0
    if total_trades:
        wins = trades.loc[trades["net_return"] > 0, "net_return"]
        losses = trades.loc[trades["net_return"] < 0, "net_return"]
        win_rate = float(len(wins) / total_trades)
        if not losses.empty:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        elif not wins.empty:
            profit_factor = float("inf")

    return {
        "model": model_name,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": float(sharpe_ratio),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "exposure": exposure,
    }
