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

## Freqtrade native runtime

Freqtrade is integrated directly from source under `trading/` without Docker.
Upstream projects are stored as Git submodules, while local runtime files live
in `trading/user_data/`.

### Trading structure

- `trading/freqtrade/` contains the Freqtrade source submodule on `stable`.
- `trading/frequi/` contains the FreqUI source submodule on `main`.
- `trading/freqtrade-strategies/` contains upstream example strategies.
- `trading/user_data/` contains local configs, strategies, data, logs, and
  backtest results.

The upstream submodules keep their own GPL-3.0 licenses. The project-level MIT
license does not relicense code inside these submodule directories.

### Clone with submodules

After cloning this repository, initialize the upstream trading sources:

```powershell
git submodule update --init --recursive
```

### Install Freqtrade

Create a dedicated Python environment inside the Freqtrade source checkout:

```powershell
cd trading/freqtrade
py -3.13 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Copy the safe config template to a local config file. The local file is ignored
by git because it can contain API keys, passwords, and private runtime settings.

```powershell
Copy-Item ..\user_data\config.example.json ..\user_data\config.json
```

### Run Freqtrade

Download historical candles for the pairs configured in
`trading/user_data/config.json`:

```powershell
freqtrade download-data --userdir ..\user_data --config ..\user_data\config.json --timeframes 5m --timerange 20250101-20250201
```

Run a backtest with a local strategy:

```powershell
freqtrade backtesting --userdir ..\user_data --config ..\user_data\config.json --strategy <StrategyClass> --strategy-path ..\user_data\strategies --timeframe 5m --timerange 20250101-20250201 --export trades
```

Backtest artifacts are written to `trading/user_data/backtest_results/` and are
intentionally ignored by git.

To test an upstream example strategy without copying it into local user data:

```powershell
freqtrade backtesting --userdir ..\user_data --config ..\user_data\config.json --strategy Strategy001 --strategy-path ..\freqtrade-strategies\user_data\strategies
```

### Run FreqUI from source

`trading/user_data/config.example.json` enables the local Freqtrade API server
on `http://127.0.0.1:8080` and allows the default Vite development origins.

Start the Freqtrade API:

```powershell
cd trading/freqtrade
. .\.venv\Scripts\Activate.ps1
freqtrade webserver --userdir ..\user_data --config ..\user_data\config.json
```

Start the FreqUI development server in another terminal:

```powershell
cd trading/frequi
corepack enable
pnpm install
pnpm run dev
```

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
