# TimeAnalysis

A Python repository for time series data analysis, documented as a blog via GitHub Pages.

## Goal

The main **goal** of this project is to build an automated trading system capable of winning in more than **60%** of cases.

## What We Are Learning

To achieve this, our team is building expertise across two domains:

**Signal & Data Skills**
- Signal processing tools for working with time series data
- ML/DL algorithms for building predictive models
- Visualization and documentation skills for exploratory data analysis (EDA)

**Market & Domain Knowledge**
- Stock exchange APIs
- Existing products and competitors in this space
- Micro and macroeconomics

 **Stochastic analysis and Probability**

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync --dev
```

This creates a `.venv` and installs all dependencies. Run once after cloning.

## Freqtrade backtesting runtime

Freqtrade is integrated as a separate Docker-based runtime under
`trading/freqtrade/`. The main Python package stays focused on reusable models
and research code, while Freqtrade acts as an execution adapter for backtests,
dry runs, and future live trading.

### Prerequisites

- Docker Desktop with Docker Compose enabled.
- A completed local Python setup via `uv sync --dev`.
- No exchange API keys are needed for the default backtesting flow.

### First-time setup

Copy the example config to a local config file. The local file is ignored by git
because it can later contain exchange keys and private runtime settings.

```powershell
cd trading/freqtrade
Copy-Item user_data\config.example.json user_data\config.json
```

```bash
cd trading/freqtrade
cp user_data/config.example.json user_data/config.json
```

Pull the Freqtrade Docker image:

```bash
docker compose pull
```

Download historical candles for the pairs configured in
`user_data/config.json`:

```bash
docker compose run --rm freqtrade download-data --config /freqtrade/user_data/config.json --timeframes 5m --timerange 20250101-20250201
```

Run a baseline backtest:

```bash
docker compose run --rm freqtrade backtesting --config /freqtrade/user_data/config.json --strategy TimeAnalysisSmaStrategy --strategy-path /freqtrade/user_data/strategies --timeframe 5m --timerange 20250101-20250201 --export trades
```

Backtest artifacts are written to
`trading/freqtrade/user_data/backtest_results/` and are intentionally ignored by
git.

### Project trading structure

- `src/time_analysis/models/` contains exchange-independent signal models.
- `trading/freqtrade/user_data/strategies/` contains thin Freqtrade adapters.
- `trading/freqtrade/user_data/config.example.json` is the safe template for
  new contributors.
- `trading/freqtrade/user_data/config.json` is local-only and should never be
  committed.

The current example uses `SmaMomentumModel`, a simple long-only moving-average
crossover model. It is intentionally simple so the full pipeline is easy to
test and inspect before adding ML or exchange-specific behavior.

### Run files

```bash
# Run a Python script
uv run python script.py

# Run Jupyter
uv run jupyter notebook

# Run tests
uv run pytest
```

No need to activate the virtual environment — `uv run` handles it automatically.

### PyCharm

Set the interpreter to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux). Jupyter notebooks will work out of the box.

## Code style: Ruff

We use [Ruff](https://docs.astral.sh/ruff/) as both linter and formatter. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

### Common commands

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix what can be fixed safely
uv run ruff check --fix .

# Format the codebase
uv run ruff format .

# Verify formatting without writing changes (used in CI)
uv run ruff format --check .
```

Run both before committing:

```bash
# bash / zsh / PowerShell 7+
uv run ruff check --fix . && uv run ruff format .
```

```powershell
# Windows PowerShell 5.1 (no && operator)
uv run ruff check --fix .; if ($?) { uv run ruff format . }
```

### CI

Every push and pull request to `main` runs Ruff via GitHub Actions
(see `.github/workflows/ruff.yml`). A PR is blocked if either lint or
format check fails — fix issues locally with the commands above and push again.

### Editor integration

- **PyCharm** — install the official *Ruff* plugin, point it at the project's
  `.venv` interpreter, and enable "Run ruff on save" and "Use ruff format".
- **VS Code** — install the *Ruff* extension by Astral; it picks up
  `pyproject.toml` automatically.

## Our Manifesto

- **Language** — We use English for all communication and documentation.
- **Code quality** — We deploy code only after it has been reviewed.
- **Product mindset** — We are building a real, commercial product — not an educational exercise.
- **AI usage** — We use AI tools only for visualization and as a reference, not for core development.
- **Documentation** — We write clear, beautiful documentation that can be understood by non-professionals.

## Research blog

Public research notebooks are published as a card-based Quarto blog at
<https://04Maksimka.github.io/TimeAnalysis/>.

To publish a new exploration, add an executed Jupyter notebook to the matching
section folder under `research/posts/` and push it to `main`. See
[`research/README.md`](research/README.md) for the notebook template, local
preview command, and the one-time GitHub Pages setting.
