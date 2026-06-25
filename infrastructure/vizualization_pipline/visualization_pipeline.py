from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib.abstract as ta

try:
    from . import custom_indicators as custom
except ImportError:  # Freqtrade loads strategy files from strategy_path as top-level modules.
    import custom_indicators as custom


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
DEFAULT_USER_DATA_DIR = PROJECT_ROOT / "trading" / "user_data"
DEFAULT_BASE_CONFIG_PATH = DEFAULT_USER_DATA_DIR / "config_for_viz.json"
SPEC_DIR = PIPELINE_DIR / "specs"
STRATEGY_NAME = "ConfigurableVisualStrategy"
SPEC_ENV_VAR = "TA_VIZ_SPEC_PATH"


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    expression: str
    panel: str = "main"
    style: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanelSpec:
    name: str


@dataclass(frozen=True)
class VisualizationRunResult:
    spec_path: Path
    plot_files: list[Path]
    dashboard_path: Path
    download_command: list[str] | None
    plot_command: list[str]
    run_name: str


def ind(
    name: str,
    expression: str,
    *,
    panel: str = "main",
    color: str | None = None,
    style: Mapping[str, Any] | None = None,
    **plot_style: Any,
) -> IndicatorSpec:
    merged_style: dict[str, Any] = {}
    if style:
        merged_style.update(dict(style))
    if color:
        merged_style["color"] = color
    merged_style.update(plot_style)
    return IndicatorSpec(name=name, expression=expression, panel=panel, style=merged_style)


def panel(name: str) -> PanelSpec:
    return PanelSpec(name=name)


def load_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_spec_from_env(*, required: bool = True) -> dict[str, Any]:
    spec_path = os.environ.get(SPEC_ENV_VAR)
    if not spec_path:
        if required:
            raise RuntimeError(f"{SPEC_ENV_VAR} is not set.")
        return {}
    return load_spec(spec_path)


def build_plot_config(spec: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    plot_config: dict[str, dict[str, dict[str, Any]]] = {
        "main_plot": {},
        "subplots": {subplot["name"]: {} for subplot in spec.get("subplots", [])},
    }
    for indicator in spec.get("indicators", []):
        indicator_name = indicator["name"]
        indicator_panel = indicator.get("panel", "main")
        indicator_style = dict(indicator.get("style", {}))
        if indicator_panel == "main":
            plot_config["main_plot"][indicator_name] = indicator_style
        else:
            plot_config["subplots"].setdefault(indicator_panel, {})[indicator_name] = (
                indicator_style
            )
    return plot_config


def apply_indicators(dataframe: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    df = dataframe
    context: dict[str, Any] = {
        "df": df,
        "ta": ta,
        "np": np,
        "pd": pd,
        "custom": custom,
    }

    for indicator in spec.get("indicators", []):
        indicator_name = indicator["name"]
        expression = indicator["expression"]
        try:
            value = eval(expression, context)  # noqa: S307 - trusted local visualization specs.
        except Exception as exc:
            raise RuntimeError(
                f"Indicator {indicator_name!r} failed while evaluating expression "
                f"{expression!r}: {exc}"
            ) from exc

        if isinstance(value, pd.DataFrame):
            if len(value.columns) != 1:
                raise RuntimeError(
                    f"Indicator {indicator_name!r} returned a DataFrame with multiple "
                    "columns. Select a single column in the expression."
                )
            value = value.iloc[:, 0]

        df[indicator_name] = value
        context["df"] = df
        context[indicator_name] = df[indicator_name]

    return df


def build_visualization(
    *,
    pairs: list[str],
    indicators: list[IndicatorSpec | Mapping[str, Any]],
    subplots: list[PanelSpec | str | Mapping[str, Any]] | None = None,
    timeframe: str = "1h",
    timerange: str | None = None,
    days: int = 365,
    plot_limit: int = 1000,
    run_name: str | None = None,
    auto_download: bool = True,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG_PATH,
    freqtrade_executable: str | Path | None = None,
) -> VisualizationRunResult:
    normalized_pairs = _normalize_pairs(pairs)
    normalized_indicators = _normalize_indicators(indicators)
    normalized_subplots = _normalize_subplots(subplots, normalized_indicators)
    normalized_run_name = _normalize_run_name(run_name)

    base_config = _resolve_project_path(base_config_path)
    user_data_dir = base_config.parent
    plot_root = user_data_dir / "plot"
    run_plot_dir = plot_root / normalized_run_name
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    spec = _build_spec(
        run_name=normalized_run_name,
        pairs=normalized_pairs,
        indicators=normalized_indicators,
        subplots=normalized_subplots,
        timeframe=timeframe,
        timerange=timerange,
        days=days,
        plot_limit=plot_limit,
    )
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = SPEC_DIR / f"{normalized_run_name}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    executable = str(freqtrade_executable or _default_freqtrade_executable())
    cwd = PROJECT_ROOT / "trading"
    env = _strategy_environment(spec_path)

    download_command = None
    if auto_download:
        download_command = _build_download_command(
            executable=executable,
            config_path=base_config,
            user_data_dir=user_data_dir,
            pairs=normalized_pairs,
            timeframe=timeframe,
            timerange=timerange,
            days=days,
        )
        _run_command(download_command, cwd=cwd, env=env)

    plot_command = _build_plot_command(
        executable=executable,
        config_path=base_config,
        user_data_dir=user_data_dir,
        pairs=normalized_pairs,
        timeframe=timeframe,
        timerange=timerange,
        plot_limit=plot_limit,
    )
    _run_command(plot_command, cwd=cwd, env=env)

    plot_files = _move_plot_files(
        pairs=normalized_pairs,
        timeframe=timeframe,
        plot_root=plot_root,
        run_plot_dir=run_plot_dir,
    )
    dashboard_path = _write_dashboard(
        run_plot_dir=run_plot_dir,
        run_name=normalized_run_name,
        pairs=normalized_pairs,
        plot_files=plot_files,
        spec=spec,
    )

    return VisualizationRunResult(
        spec_path=spec_path,
        plot_files=plot_files,
        dashboard_path=dashboard_path,
        download_command=download_command,
        plot_command=plot_command,
        run_name=normalized_run_name,
    )


def _normalize_pairs(pairs: Sequence[str]) -> list[str]:
    normalized = [pair.strip() for pair in pairs if pair and pair.strip()]
    if not normalized:
        raise ValueError("At least one pair is required.")
    return normalized


def _normalize_indicators(
    indicators: Sequence[IndicatorSpec | Mapping[str, Any]],
) -> list[IndicatorSpec]:
    normalized: list[IndicatorSpec] = []
    names: set[str] = set()
    for item in indicators:
        if isinstance(item, IndicatorSpec):
            spec = item
        else:
            spec = IndicatorSpec(
                name=str(item["name"]),
                expression=str(item["expression"]),
                panel=str(item.get("panel", "main")),
                style=dict(item.get("style", {})),
            )
        if not spec.name:
            raise ValueError("Indicator name cannot be empty.")
        if spec.name in names:
            raise ValueError(f"Duplicate indicator name: {spec.name}")
        if not spec.expression:
            raise ValueError(f"Indicator {spec.name!r} expression cannot be empty.")
        if not spec.panel:
            raise ValueError(f"Indicator {spec.name!r} panel cannot be empty.")
        names.add(spec.name)
        normalized.append(spec)
    return normalized


def _normalize_subplots(
    subplots: Sequence[PanelSpec | str | Mapping[str, Any]] | None,
    indicators: Sequence[IndicatorSpec],
) -> list[PanelSpec]:
    if subplots is None:
        panel_names: list[str] = []
        for indicator in indicators:
            if indicator.panel != "main" and indicator.panel not in panel_names:
                panel_names.append(indicator.panel)
        return [PanelSpec(name=name) for name in panel_names]

    normalized: list[PanelSpec] = []
    names: set[str] = set()
    for item in subplots:
        if isinstance(item, PanelSpec):
            subplot = item
        elif isinstance(item, str):
            subplot = PanelSpec(name=item)
        else:
            subplot = PanelSpec(name=str(item["name"]))
        if not subplot.name:
            raise ValueError("Subplot name cannot be empty.")
        if subplot.name == "main":
            raise ValueError("'main' is reserved for the primary candlestick panel.")
        if subplot.name in names:
            raise ValueError(f"Duplicate subplot name: {subplot.name}")
        names.add(subplot.name)
        normalized.append(subplot)

    missing = sorted(
        {indicator.panel for indicator in indicators if indicator.panel != "main"} - names
    )
    if missing:
        raise ValueError(f"Indicator panels are not declared in subplots: {missing}")
    return normalized


def _normalize_run_name(run_name: str | None) -> str:
    raw_name = run_name or datetime.now(UTC).strftime("viz_%Y%m%d_%H%M%S")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("run_name cannot be empty after normalization.")
    return normalized


def _build_spec(
    *,
    run_name: str,
    pairs: list[str],
    indicators: list[IndicatorSpec],
    subplots: list[PanelSpec],
    timeframe: str,
    timerange: str | None,
    days: int,
    plot_limit: int,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "pairs": pairs,
        "timeframe": timeframe,
        "timerange": timerange,
        "days": days,
        "plot_limit": plot_limit,
        "startup_candle_count": 100,
        "indicators": [
            {
                "name": indicator.name,
                "expression": indicator.expression,
                "panel": indicator.panel,
                "style": dict(indicator.style),
            }
            for indicator in indicators
        ],
        "subplots": [{"name": subplot.name} for subplot in subplots],
    }


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _default_freqtrade_executable() -> Path | str:
    candidates = [
        PROJECT_ROOT / "trading" / ".venv" / "Scripts" / "freqtrade.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "freqtrade.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return "freqtrade"


def _strategy_environment(spec_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env[SPEC_ENV_VAR] = str(spec_path)
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(PIPELINE_DIR), str(PROJECT_ROOT)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _build_download_command(
    *,
    executable: str,
    config_path: Path,
    user_data_dir: Path,
    pairs: list[str],
    timeframe: str,
    timerange: str | None,
    days: int,
) -> list[str]:
    command = [
        executable,
        "download-data",
        "-c",
        str(config_path),
        "--userdir",
        str(user_data_dir),
        "--pairs",
        *pairs,
        "--timeframes",
        timeframe,
    ]
    if timerange:
        command.extend(["--timerange", timerange, "--prepend"])
    else:
        command.extend(["--days", str(days)])
    return command


def _build_plot_command(
    *,
    executable: str,
    config_path: Path,
    user_data_dir: Path,
    pairs: list[str],
    timeframe: str,
    timerange: str | None,
    plot_limit: int,
) -> list[str]:
    command = [
        executable,
        "plot-dataframe",
        "-c",
        str(config_path),
        "--userdir",
        str(user_data_dir),
        "--strategy",
        STRATEGY_NAME,
        "--strategy-path",
        str(PIPELINE_DIR),
        "-p",
        *pairs,
        "-i",
        timeframe,
        "--no-trades",
        "--plot-limit",
        str(plot_limit),
    ]
    if timerange:
        command.extend(["--timerange", timerange])
    return command


def _run_command(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=dict(env), check=True)


def _move_plot_files(
    *,
    pairs: list[str],
    timeframe: str,
    plot_root: Path,
    run_plot_dir: Path,
) -> list[Path]:
    plot_files: list[Path] = []
    for pair in pairs:
        source = plot_root / f"freqtrade-plot-{_pair_to_filename(pair)}-{timeframe}.html"
        if not source.exists():
            raise FileNotFoundError(f"Expected plot file was not generated: {source}")
        destination = run_plot_dir / source.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
        plot_files.append(destination)
    return plot_files


def _pair_to_filename(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def _write_dashboard(
    *,
    run_plot_dir: Path,
    run_name: str,
    pairs: list[str],
    plot_files: list[Path],
    spec: Mapping[str, Any],
) -> Path:
    plots = [
        {
            "pair": pair,
            "file": plot_file.name,
        }
        for pair, plot_file in zip(pairs, plot_files, strict=True)
    ]
    first_file = plots[0]["file"] if plots else ""
    buttons = "\n".join(
        (
            f'<button class="pair-button" data-file="{escape(plot["file"])}">'
            f"{escape(plot['pair'])}</button>"
        )
        for plot in plots
    )
    indicators = ", ".join(indicator["name"] for indicator in spec.get("indicators", []))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(run_name)} visualization</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #f6f7f9;
      color: #1f2937;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
    }}
    aside {{
      border-right: 1px solid #d6dae1;
      background: #ffffff;
      padding: 18px;
      overflow-y: auto;
    }}
    main {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      background: #eef1f5;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.25;
      margin: 0 0 8px;
    }}
    .meta {{
      color: #5b6472;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 18px;
      overflow-wrap: anywhere;
    }}
    .pair-button {{
      width: 100%;
      min-height: 38px;
      border: 1px solid #d6dae1;
      background: #ffffff;
      color: #1f2937;
      border-radius: 6px;
      margin-bottom: 8px;
      text-align: left;
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
    }}
    .pair-button:hover,
    .pair-button.active {{
      border-color: #2563eb;
      background: #eff6ff;
    }}
    .toolbar {{
      min-height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      background: #ffffff;
      border-bottom: 1px solid #d6dae1;
    }}
    .toolbar-title {{
      font-weight: 600;
    }}
    .toolbar a {{
      color: #2563eb;
      text-decoration: none;
      font-size: 14px;
    }}
    iframe {{
      flex: 1;
      width: 100%;
      border: 0;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <aside>
    <h1>{escape(run_name)}</h1>
    <div class="meta">
      Timeframe: {escape(str(spec.get("timeframe", "")))}<br>
      Timerange: {escape(str(spec.get("timerange") or "latest downloaded data"))}<br>
      Indicators: {escape(indicators or "none")}
    </div>
    {buttons}
  </aside>
  <main>
    <div class="toolbar">
      <div class="toolbar-title" id="currentPair"></div>
      <a id="openLink" href="{escape(first_file)}" target="_blank" rel="noopener">Open graph</a>
    </div>
    <iframe id="plotFrame" src="{escape(first_file)}" title="Freqtrade plot"></iframe>
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll(".pair-button"));
    const frame = document.getElementById("plotFrame");
    const openLink = document.getElementById("openLink");
    const currentPair = document.getElementById("currentPair");

    function selectButton(button) {{
      buttons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      frame.src = button.dataset.file;
      openLink.href = button.dataset.file;
      currentPair.textContent = button.textContent;
    }}

    buttons.forEach((button) => {{
      button.addEventListener("click", () => selectButton(button));
    }});
    if (buttons.length > 0) {{
      selectButton(buttons[0]);
    }}
  </script>
</body>
</html>
"""
    dashboard_path = run_plot_dir / "index.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
