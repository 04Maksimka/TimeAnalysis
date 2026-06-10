"""Helpers for turning Freqtrade backtest artifacts into report tables."""

from __future__ import annotations

import json
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def find_latest_backtest_result(results_dir: Path | str) -> Path:
    """Find the latest Freqtrade backtest result file.

    Uses Freqtrade's ``.last_result.json`` pointer when it is present, then
    falls back to the newest JSON or ZIP result file in the directory.

    :param results_dir: directory containing Freqtrade backtest artifacts
    :return: path to the newest main backtest result file
    """

    directory = Path(results_dir)
    last_result = directory / ".last_result.json"
    if last_result.exists():
        latest_name = json.loads(last_result.read_text(encoding="utf-8")).get(
            "latest_backtest"
        )
        if latest_name:
            latest_path = directory / latest_name
            if latest_path.exists():
                return latest_path

    candidates = [
        path
        for path in directory.glob("backtest-result-*")
        if path.is_file()
        and path.suffix in {".json", ".zip"}
        and not path.name.endswith(".meta.json")
        and "_config" not in path.name
    ]
    if not candidates:
        msg = f"No Freqtrade backtest result files found in {directory}"
        raise FileNotFoundError(msg)

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_backtest_payload(result_path: Path | str) -> dict[str, Any]:
    """Load the main JSON payload from a Freqtrade result file.

    :param result_path: path to a Freqtrade JSON or ZIP result artifact
    :return: decoded Freqtrade result payload
    """

    path = Path(result_path)
    if path.suffix == ".zip":
        return _load_backtest_payload_from_zip(path)

    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_backtest_report(
    results_dir: Path | str,
    strategy_name: str | None = None,
) -> FreqtradeBacktestReport:
    """Load the newest backtest result as a report object.

    :param results_dir: directory containing Freqtrade backtest artifacts
    :param strategy_name: optional strategy name to select from the payload
    :return: report object with normalized metric and table helpers
    """

    source_path = find_latest_backtest_result(results_dir)
    payload = load_backtest_payload(source_path)
    selected_strategy_name, strategy_payload = _select_strategy_payload(
        payload,
        strategy_name,
    )
    return FreqtradeBacktestReport(
        source_path=source_path,
        strategy_name=selected_strategy_name,
        payload=strategy_payload,
    )


@dataclass(frozen=True, slots=True)
class FreqtradeBacktestReport:
    """Report-friendly view of one Freqtrade strategy backtest result.

    Attributes:
        source_path: path to the source Freqtrade result artifact
        strategy_name: selected strategy name from the Freqtrade payload
        payload: strategy-specific Freqtrade result payload
    """

    source_path: Path
    strategy_name: str
    payload: dict[str, Any]

    def metrics(self) -> dict[str, Any]:
        """Return headline metrics in raw numeric form.

        :return: dictionary of strategy, period, trade, profit, and drawdown metrics
        """

        return {
            "strategy": self.strategy_name,
            "source_file": self.source_path.name,
            "backtest_start": self.payload.get("backtest_start"),
            "backtest_end": self.payload.get("backtest_end"),
            "total_trades": self.payload.get("total_trades", 0),
            "wins": self.payload.get("wins", 0),
            "draws": self.payload.get("draws", 0),
            "losses": self.payload.get("losses", 0),
            "winrate": self.payload.get("winrate", 0.0),
            "profit_total": self.payload.get("profit_total", 0.0),
            "profit_total_abs": self.payload.get("profit_total_abs", 0.0),
            "profit_factor": self.payload.get("profit_factor"),
            "expectancy": self.payload.get("expectancy"),
            "max_drawdown_account": self.payload.get("max_drawdown_account", 0.0),
            "max_drawdown_abs": self.payload.get("max_drawdown_abs", 0.0),
            "final_balance": self.payload.get("final_balance"),
            "dry_run_wallet": self.payload.get("dry_run_wallet"),
            "best_pair": self.payload.get("best_pair"),
        }

    def trades_frame(self) -> pd.DataFrame:
        """Return normalized trade rows sorted by close time.

        :return: dataframe of trades with parsed dates and numeric columns
        """

        trades = pd.DataFrame(self.payload.get("trades", []))
        if trades.empty:
            return trades

        for column in ["open_date", "close_date"]:
            if column in trades:
                trades[column] = pd.to_datetime(trades[column], utc=True)

        for column in ["open_timestamp", "close_timestamp"]:
            if column in trades:
                trades[column] = pd.to_datetime(
                    trades[column],
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

        for column in [
            "amount",
            "open_rate",
            "close_rate",
            "profit_abs",
            "profit_ratio",
            "stake_amount",
            "trade_duration",
        ]:
            if column in trades:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")

        sort_column = "close_date" if "close_date" in trades else "close_timestamp"
        if sort_column in trades:
            trades = trades.sort_values(sort_column).reset_index(drop=True)

        return trades

    def pair_summary_frame(self) -> pd.DataFrame:
        """Return Freqtrade's per-pair summary as a dataframe.

        :return: dataframe with one row per traded pair
        """

        frame = pd.DataFrame(self.payload.get("results_per_pair", []))
        if frame.empty:
            return frame

        frame = frame.rename(columns={"key": "pair"})
        return _coerce_numeric_columns(frame)

    def exit_reason_summary_frame(self) -> pd.DataFrame:
        """Return Freqtrade's exit reason summary as a dataframe.

        :return: dataframe with one row per Freqtrade exit reason
        """

        frame = pd.DataFrame(self.payload.get("exit_reason_summary", []))
        if frame.empty:
            return frame

        frame = frame.rename(columns={"key": "exit_reason"})
        return _coerce_numeric_columns(frame)

    def daily_profit_frame(self) -> pd.DataFrame:
        """Return daily absolute profit with cumulative profit.

        :return: dataframe with daily and cumulative absolute profit
        """

        periodic = self.payload.get("periodic_breakdown", {})
        if isinstance(periodic, dict) and periodic.get("day"):
            frame = pd.DataFrame(periodic["day"])
            if "date_ts" in frame:
                frame["date"] = pd.to_datetime(frame["date_ts"], unit="ms", utc=True)
            else:
                frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, utc=True)
            frame = _coerce_numeric_columns(frame)
            frame["cumulative_profit_abs"] = frame["profit_abs"].cumsum()
            return frame.sort_values("date").reset_index(drop=True)

        rows = self.payload.get("daily_profit", [])
        frame = pd.DataFrame(rows, columns=["date", "profit_abs"])
        if frame.empty:
            return frame

        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["profit_abs"] = pd.to_numeric(frame["profit_abs"], errors="coerce")
        frame["cumulative_profit_abs"] = frame["profit_abs"].cumsum()
        return frame.sort_values("date").reset_index(drop=True)

    def equity_curve_frame(self) -> pd.DataFrame:
        """Return a trade-by-trade equity and drawdown curve.

        :return: dataframe with equity, peak equity, and drawdown after each trade
        """

        trades = self.trades_frame()
        if trades.empty or "profit_abs" not in trades:
            return pd.DataFrame()

        time_column = "close_date" if "close_date" in trades else "close_timestamp"
        initial_balance = self.payload.get("dry_run_wallet") or 0.0

        curve = trades[[time_column, "pair", "profit_abs", "profit_ratio"]].copy()
        curve = curve.rename(columns={time_column: "time"})
        curve["trade_number"] = range(1, len(curve) + 1)
        curve["cumulative_profit_abs"] = curve["profit_abs"].cumsum()
        curve["equity"] = initial_balance + curve["cumulative_profit_abs"]
        curve["equity_peak"] = curve["equity"].cummax()
        curve["drawdown_abs"] = curve["equity_peak"] - curve["equity"]
        curve["drawdown_ratio"] = curve["drawdown_abs"] / curve["equity_peak"]
        return curve


def _load_backtest_payload_from_zip(path: Path) -> dict[str, Any]:
    """Load the main JSON payload from a zipped Freqtrade artifact.

    :param path: path to a Freqtrade ZIP result artifact
    :return: decoded main JSON payload from the archive
    """

    with zipfile.ZipFile(path) as archive:
        json_names = [
            name
            for name in archive.namelist()
            if name.endswith(".json")
            and not name.endswith(".meta.json")
            and "_config" not in name
        ]
        if not json_names:
            msg = f"No main Freqtrade JSON payload found in {path}"
            raise FileNotFoundError(msg)

        return json.loads(archive.read(json_names[0]).decode("utf-8"))


def _select_strategy_payload(
    payload: dict[str, Any],
    strategy_name: str | None,
) -> tuple[str, dict[str, Any]]:
    """Select one strategy payload from a Freqtrade result payload.

    :param payload: decoded top-level Freqtrade result payload
    :param strategy_name: optional strategy name to select
    :return: selected strategy name and strategy-specific payload
    """

    strategies = payload.get("strategy")
    if not isinstance(strategies, dict) or not strategies:
        msg = "Freqtrade payload does not contain a non-empty 'strategy' object"
        raise ValueError(msg)

    if strategy_name is None:
        selected_name = next(iter(strategies))
        return selected_name, strategies[selected_name]

    if strategy_name not in strategies:
        available = ", ".join(strategies)
        msg = f"Strategy {strategy_name!r} not found. Available strategies: {available}"
        raise KeyError(msg)

    return strategy_name, strategies[strategy_name]


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce report table columns to numeric types where possible.

    :param frame: dataframe returned from a Freqtrade summary section
    :return: dataframe copy with numeric columns converted
    """

    result = frame.copy()
    for column in result.columns:
        if column not in {"pair", "exit_reason", "duration_avg", "key", "date"}:
            with suppress(TypeError, ValueError):
                result[column] = pd.to_numeric(result[column], errors="raise")
    return result
