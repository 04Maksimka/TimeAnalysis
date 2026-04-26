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
