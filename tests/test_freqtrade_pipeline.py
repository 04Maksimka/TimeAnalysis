from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from time_analysis.backtesting.freqtrade_pipeline import (
    BacktestExperimentConfig,
    FreqtradePipelineError,
    build_backtesting_command,
    build_download_data_command,
    ensure_freqtrade_config,
    run_checked_freqtrade_command,
    run_freqtrade_command,
)


def test_build_download_data_command_uses_experiment_values() -> None:
    """Check that download-data command uses experiment timeframe and timerange."""

    config = BacktestExperimentConfig(
        strategy="DemoStrategy",
        timeframe="15m",
        timerange="20250101-20250201",
    )

    assert build_download_data_command(config) == (
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "download-data",
        "--config",
        "/freqtrade/user_data/config.json",
        "--timeframes",
        "15m",
        "--timerange",
        "20250101-20250201",
    )


def test_build_backtesting_command_exports_trades_by_default() -> None:
    """Check that backtesting command exports trades and preserves extra args."""

    config = BacktestExperimentConfig(
        strategy="DemoStrategy",
        timeframe="1h",
        timerange="20240101-20240201",
        extra_backtesting_args=("--enable-protections",),
    )

    assert build_backtesting_command(config) == (
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "backtesting",
        "--config",
        "/freqtrade/user_data/config.json",
        "--strategy",
        "DemoStrategy",
        "--strategy-path",
        "/freqtrade/user_data/strategies",
        "--timeframe",
        "1h",
        "--timerange",
        "20240101-20240201",
        "--export",
        "trades",
        "--enable-protections",
    )


def test_ensure_freqtrade_config_copies_example(tmp_path: Path) -> None:
    """Check that missing local config is copied from the example config.

    :param tmp_path: pytest temporary directory fixture
    """

    freqtrade_dir = tmp_path / "freqtrade"
    user_data_dir = freqtrade_dir / "user_data"
    user_data_dir.mkdir(parents=True)
    example_config = user_data_dir / "config.example.json"
    example_config.write_text('{"dry_run": true}', encoding="utf-8")

    config = BacktestExperimentConfig(freqtrade_dir=freqtrade_dir)

    config_path = ensure_freqtrade_config(config)

    assert config_path == user_data_dir / "config.json"
    assert config_path.read_text(encoding="utf-8") == '{"dry_run": true}'


def test_run_checked_freqtrade_command_returns_success(tmp_path: Path) -> None:
    """Check that successful command results are returned unchanged.

    :param tmp_path: pytest temporary directory fixture
    """

    completed = subprocess.CompletedProcess(
        args=("docker",),
        returncode=0,
        stdout="ok",
        stderr="",
    )

    def runner(
        command: Sequence[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Validate command inputs and return a successful completed process.

        :param command: command arguments passed by the pipeline
        :param cwd: working directory passed by the pipeline
        :return: successful completed process fixture
        """

        assert command == ("docker", "compose")
        assert cwd == tmp_path
        return completed

    assert run_checked_freqtrade_command(("docker", "compose"), tmp_path, runner)


def test_run_freqtrade_command_uses_utf8_with_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that subprocess output decoding is stable on Windows.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """

    captured_kwargs = {}
    completed = subprocess.CompletedProcess(
        args=("docker",),
        returncode=0,
        stdout="ok",
        stderr="",
    )

    def fake_run(
        command: list[str],
        **kwargs,
    ) -> subprocess.CompletedProcess[str]:
        """Capture subprocess keyword arguments and return success.

        :param command: command arguments passed to ``subprocess.run``
        :param kwargs: subprocess keyword arguments
        :return: successful completed process fixture
        """

        captured_kwargs.update(kwargs)
        return completed

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_freqtrade_command(("docker", "compose"), tmp_path)

    assert result is completed
    assert captured_kwargs["encoding"] == "utf-8"
    assert captured_kwargs["errors"] == "replace"


def test_run_checked_freqtrade_command_raises_readable_error(tmp_path: Path) -> None:
    """Check that failed command output is included in the pipeline error.

    :param tmp_path: pytest temporary directory fixture
    """

    completed = subprocess.CompletedProcess(
        args=("docker",),
        returncode=2,
        stdout="download failed",
        stderr="bad config",
    )

    def runner(
        command: Sequence[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Return a failed completed process fixture.

        :param command: command arguments passed by the pipeline
        :param cwd: working directory passed by the pipeline
        :return: failed completed process fixture
        """

        return completed

    with pytest.raises(FreqtradePipelineError) as exc_info:
        run_checked_freqtrade_command(("docker", "compose"), tmp_path, runner)

    message = str(exc_info.value)
    assert "exit code 2" in message
    assert "download failed" in message
    assert "bad config" in message
