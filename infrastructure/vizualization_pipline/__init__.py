"""Configurable Freqtrade visualization pipeline."""

try:
    from .visualization_pipeline import (
        IndicatorSpec,
        PanelSpec,
        VisualizationRunResult,
        build_plot_config,
        build_visualization,
        ind,
        panel,
    )
except ImportError:  # Freqtrade may import this folder directly as a strategy_path.
    from visualization_pipeline import (
        IndicatorSpec,
        PanelSpec,
        VisualizationRunResult,
        build_plot_config,
        build_visualization,
        ind,
        panel,
    )

__all__ = [
    "IndicatorSpec",
    "PanelSpec",
    "VisualizationRunResult",
    "build_plot_config",
    "build_visualization",
    "ind",
    "panel",
]
