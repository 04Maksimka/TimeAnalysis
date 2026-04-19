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

## Our Manifesto

- **Language** — We use English for all communication and documentation.
- **Code quality** — We deploy code only after it has been reviewed.
- **Product mindset** — We are building a real, commercial product — not an educational exercise.
- **AI usage** — We use AI tools only for visualization and as a reference, not for core development.
- **Documentation** — We write clear, beautiful documentation that can be understood by non-professionals.
