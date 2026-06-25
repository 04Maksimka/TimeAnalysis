from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from infrastructure.vizualization_pipline import (
    build_plot_config,
    build_visualization,
    ind,
    panel,
)
from infrastructure.vizualization_pipline.visualization_pipeline import apply_indicators


def test_plot_config_preserves_subplot_order() -> None:
    spec = {
        "indicators": [
            {"name": "sma_20", "expression": "df['close']", "panel": "main", "style": {}},
            {
                "name": "rsi",
                "expression": "df['close']",
                "panel": "RSI",
                "style": {"color": "#7c3aed"},
            },
            {
                "name": "macd",
                "expression": "df['close']",
                "panel": "MACD",
                "style": {},
            },
        ],
        "subplots": [{"name": "RSI"}, {"name": "MACD"}],
    }

    plot_config = build_plot_config(spec)

    assert list(plot_config["subplots"]) == ["RSI", "MACD"]
    assert "sma_20" in plot_config["main_plot"]
    assert plot_config["subplots"]["RSI"]["rsi"]["color"] == "#7c3aed"


def test_apply_indicators_supports_df_ta_np_pd_and_custom() -> None:
    df = pd.DataFrame(
        {
            "open": range(1, 31),
            "high": range(2, 32),
            "low": range(0, 30),
            "close": range(1, 31),
            "volume": range(10, 40),
        }
    )
    spec = {
        "indicators": [
            {"name": "pct", "expression": "df['close'].pct_change()", "panel": "main"},
            {"name": "log_close", "expression": "np.log(df['close'])", "panel": "main"},
            {"name": "sma_5", "expression": "ta.SMA(df, timeperiod=5)", "panel": "main"},
            {
                "name": "typical",
                "expression": "custom.typical_price(df)",
                "panel": "main",
            },
        ]
    }

    result = apply_indicators(df, spec)

    assert {"pct", "log_close", "sma_5", "typical"}.issubset(result.columns)
    assert result["typical"].iloc[-1] == pytest.approx((31 + 29 + 30) / 3)


def test_apply_indicators_fails_fast_with_indicator_name() -> None:
    df = pd.DataFrame({"close": [1, 2, 3]})
    spec = {
        "indicators": [
            {"name": "broken", "expression": "df['missing']", "panel": "main"},
        ]
    }

    with pytest.raises(RuntimeError, match="broken"):
        apply_indicators(df, spec)


def test_build_visualization_writes_spec_and_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config_for_viz.json"
    user_data = tmp_path
    plot_root = user_data / "plot"
    config_path.write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        commands.append(command)
        if "plot-dataframe" in command:
            plot_root.mkdir(parents=True, exist_ok=True)
            for pair in ["BTC/USDT", "ETH/USDT"]:
                filename = f"freqtrade-plot-{pair.replace('/', '_')}-1h.html"
                (plot_root / filename).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(
        "infrastructure.vizualization_pipline.visualization_pipeline._run_command",
        fake_run,
    )
    monkeypatch.setattr(
        "infrastructure.vizualization_pipline.visualization_pipeline.SPEC_DIR",
        tmp_path / "specs",
    )

    result = build_visualization(
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        timerange="20250601-20260625",
        indicators=[
            ind("sma_20", "ta.SMA(df, timeperiod=20)", panel="main"),
            ind("rsi", "ta.RSI(df, timeperiod=14)", panel="RSI"),
        ],
        subplots=[panel("RSI")],
        run_name="pytest_run",
        base_config_path=config_path,
        freqtrade_executable="freqtrade",
    )

    spec = json.loads(result.spec_path.read_text(encoding="utf-8"))

    assert spec["pairs"] == ["BTC/USDT", "ETH/USDT"]
    assert spec["timeframe"] == "1h"
    assert spec["timerange"] == "20250601-20260625"
    assert "--timerange" in result.download_command
    assert "--prepend" in result.download_command
    assert result.dashboard_path.exists()
    assert len(result.plot_files) == 2
    assert commands[0][1] == "download-data"
    assert commands[1][1] == "plot-dataframe"


def test_build_visualization_uses_days_when_timerange_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config_for_viz.json"
    plot_root = tmp_path / "plot"
    config_path.write_text("{}", encoding="utf-8")

    def fake_run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        if "plot-dataframe" in command:
            plot_root.mkdir(parents=True, exist_ok=True)
            (plot_root / "freqtrade-plot-BTC_USDT-1d.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "infrastructure.vizualization_pipline.visualization_pipeline._run_command",
        fake_run,
    )
    monkeypatch.setattr(
        "infrastructure.vizualization_pipline.visualization_pipeline.SPEC_DIR",
        tmp_path / "specs",
    )

    result = build_visualization(
        pairs=["BTC/USDT"],
        timeframe="1d",
        indicators=[],
        days=30,
        run_name="days_run",
        base_config_path=config_path,
        freqtrade_executable="freqtrade",
    )

    assert result.download_command is not None
    assert "--days" in result.download_command
    assert "30" in result.download_command
