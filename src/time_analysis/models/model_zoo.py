"""Factory helpers for standard TimeAnalysis model candidates."""

from __future__ import annotations

from time_analysis.models.base import SignalModel
from time_analysis.models.financial_math import (
    AtrVolatilityBreakoutModel,
    BollingerRsiMeanReversionModel,
    BuyAndHoldBenchmarkModel,
    DonchianBreakoutModel,
    EmaTrendModel,
    MacdRsiTrendModel,
    WarmupMomentumHoldModel,
)
from time_analysis.models.ml_direction import (
    HistGradientBoostingDirectionModel,
    RandomForestDirectionModel,
)
from time_analysis.models.simple_momentum import SmaMomentumModel


def default_model_zoo(include_ml: bool = True) -> tuple[SignalModel, ...]:
    """Create the default set of comparable signal models.

    :param include_ml: include slower walk-forward scikit-learn models
    :return: tuple of model instances with the shared ``SignalModel`` interface
    """

    models: list[SignalModel] = [
        BuyAndHoldBenchmarkModel(),
        WarmupMomentumHoldModel(),
        SmaMomentumModel(fast_window=12, slow_window=48),
        EmaTrendModel(fast_window=12, slow_window=48),
        DonchianBreakoutModel(entry_window=55, exit_window=20),
        BollingerRsiMeanReversionModel(),
        MacdRsiTrendModel(),
        AtrVolatilityBreakoutModel(),
    ]
    if include_ml:
        models.extend(
            [
                RandomForestDirectionModel(
                    lookback_window=1500,
                    min_train_size=500,
                    retrain_interval=168,
                    n_estimators=40,
                    max_depth=4,
                    min_samples_leaf=30,
                ),
                HistGradientBoostingDirectionModel(
                    lookback_window=1500,
                    min_train_size=500,
                    retrain_interval=168,
                    max_iter=60,
                    max_leaf_nodes=10,
                ),
            ]
        )
    return tuple(models)
