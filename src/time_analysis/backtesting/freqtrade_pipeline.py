"""Executable Freqtrade backtesting pipeline for TimeAnalysis models."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from time_analysis.backtesting.freqtrade_report import (
    FreqtradeBacktestReport,
    load_latest_backtest_report,
)

CONTAINER_CONFIG_PATH = "/freqtrade/user_data/config.json"
CONTAINER_STRATEGY_PATH = "/freqtrade/user_data/strategies"

CommandRunner = Callable[
    [Sequence[str], Path],
    subprocess.CompletedProcess[str],
]


class FreqtradePipelineError(RuntimeError):
    """Error raised when a Freqtrade pipeline command fails."""


def default_project_root() -> Path:
    """Return the repository root for the installed source tree.

    :return: repository root path resolved from the package file location
    """

    return Path(__file__).resolve().parents[3]


def default_freqtrade_dir() -> Path:
    """Return the default Docker-based Freqtrade runtime directory.

    :return: path to ``trading/freqtrade`` inside the repository
    """

    return default_project_root() / "trading" / "freqtrade"


@dataclass(frozen=True, slots=True)
class BacktestExperimentConfig:
    """Configuration for one reproducible Freqtrade backtest experiment.

    Attributes:
        name: human-readable experiment name for notebooks and logs
        strategy: Freqtrade strategy class name to run
        timeframe: candle timeframe passed to Freqtrade
        timerange: Freqtrade timerange string, for example ``20250101-20250201``
        freqtrade_dir: local Docker Compose runtime directory
        config_path: optional local Freqtrade config path
        config_example_path: optional template config path
        results_dir: optional Freqtrade backtest results directory
        download_data: download candles before running the backtest
        export_trades: export trade rows into the backtest artifact
        extra_backtesting_args: additional raw arguments for Freqtrade backtesting
    """

    name: str = "sma_baseline"
    strategy: str = "TimeAnalysisSmaStrategy"
    timeframe: str = "5m"
    timerange: str = "20250101-20250201"
    freqtrade_dir: Path | str = field(default_factory=default_freqtrade_dir)
    config_path: Path | str | None = None
    config_example_path: Path | str | None = None
    results_dir: Path | str | None = None
    download_data: bool = True
    export_trades: bool = True
    extra_backtesting_args: tuple[str, ...] = ()

    @property
    def resolved_freqtrade_dir(self) -> Path:
        """Return ``freqtrade_dir`` as a ``Path``.

        :return: normalized local Freqtrade runtime directory
        """

        return Path(self.freqtrade_dir)

    @property
    def resolved_user_data_dir(self) -> Path:
        """Return the Freqtrade ``user_data`` directory.

        :return: path to the local ``user_data`` directory
        """

        return self.resolved_freqtrade_dir / "user_data"

    @property
    def resolved_config_path(self) -> Path:
        """Return the local config file used by Freqtrade.

        :return: explicit ``config_path`` or default ``user_data/config.json``
        """

        if self.config_path is not None:
            return Path(self.config_path)
        return self.resolved_user_data_dir / "config.json"

    @property
    def resolved_config_example_path(self) -> Path:
        """Return the template config copied on first run.

        :return: explicit template path or default ``user_data/config.example.json``
        """

        if self.config_example_path is not None:
            return Path(self.config_example_path)
        return self.resolved_user_data_dir / "config.example.json"

    @property
    def resolved_results_dir(self) -> Path:
        """Return the directory containing Freqtrade backtest artifacts.

        :return: explicit results directory or default ``user_data/backtest_results``
        """

        if self.results_dir is not None:
            return Path(self.results_dir)
        return self.resolved_user_data_dir / "backtest_results"


def ensure_freqtrade_config(config: BacktestExperimentConfig) -> Path:
    """Ensure that the local ignored Freqtrade config exists.

    Copies ``config.example.json`` to ``config.json`` when the local config is
    missing. Existing local configs are left untouched.

    :param config: experiment configuration with Freqtrade paths
    :return: path to the local config file
    """

    config_path = config.resolved_config_path
    if config_path.exists():
        return config_path

    example_path = config.resolved_config_example_path
    if not example_path.exists():
        msg = f"Freqtrade config example does not exist: {example_path}"
        raise FileNotFoundError(msg)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example_path, config_path)
    return config_path


def build_download_data_command(config: BacktestExperimentConfig) -> tuple[str, ...]:
    """Build the Docker Compose command that downloads historical candles.

    :param config: experiment configuration with timeframe and timerange
    :return: command arguments suitable for ``subprocess.run``
    """

    return (
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "download-data",
        "--config",
        CONTAINER_CONFIG_PATH,
        "--timeframes",
        config.timeframe,
        "--timerange",
        config.timerange,
    )


def build_backtesting_command(config: BacktestExperimentConfig) -> tuple[str, ...]:
    """Build the Docker Compose command that runs one Freqtrade backtest.

    :param config: experiment configuration with strategy and backtest settings
    :return: command arguments suitable for ``subprocess.run``
    """

    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "backtesting",
        "--config",
        CONTAINER_CONFIG_PATH,
        "--strategy",
        config.strategy,
        "--strategy-path",
        CONTAINER_STRATEGY_PATH,
        "--timeframe",
        config.timeframe,
        "--timerange",
        config.timerange,
    ]
    if config.export_trades:
        command.extend(["--export", "trades"])
    command.extend(config.extra_backtesting_args)
    return tuple(command)


def run_freqtrade_command(
    command: Sequence[str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run a Freqtrade command and capture its output.

    :param command: command arguments passed to ``subprocess.run``
    :param cwd: working directory for Docker Compose
    :return: completed subprocess with captured text output
    """

    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_checked_freqtrade_command(
    command: Sequence[str],
    cwd: Path,
    runner: CommandRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Freqtrade command and raise a readable error on failure.

    :param command: command arguments passed to the selected runner
    :param cwd: working directory for Docker Compose
    :param runner: optional command runner used by tests or custom execution
    :return: successful completed subprocess
    """

    selected_runner = runner or run_freqtrade_command
    completed = selected_runner(command, cwd)
    if completed.returncode == 0:
        return completed

    rendered_command = " ".join(command)
    msg = (
        f"Freqtrade command failed with exit code {completed.returncode}: "
        f"{rendered_command}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n"
        f"{completed.stderr}"
    )
    raise FreqtradePipelineError(msg)


def run_freqtrade_backtest(
    config: BacktestExperimentConfig | None = None,
    runner: CommandRunner | None = None,
) -> FreqtradeBacktestReport:
    """Download data if requested, run a backtest, and load its latest report.

    :param config: experiment configuration; defaults to ``BacktestExperimentConfig``
    :param runner: optional command runner used by tests or custom execution
    :return: parsed report for the newest Freqtrade backtest result
    """

    experiment = config or BacktestExperimentConfig()
    ensure_freqtrade_config(experiment)

    if experiment.download_data:
        run_checked_freqtrade_command(
            build_download_data_command(experiment),
            experiment.resolved_freqtrade_dir,
            runner,
        )

    run_checked_freqtrade_command(
        build_backtesting_command(experiment),
        experiment.resolved_freqtrade_dir,
        runner,
    )
    return load_latest_backtest_report(
        experiment.resolved_results_dir,
        strategy_name=experiment.strategy,
    )
