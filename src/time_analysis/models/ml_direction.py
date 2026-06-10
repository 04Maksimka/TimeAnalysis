"""Walk-forward machine-learning direction models for candle data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from time_analysis.models.base import (
    ENTRY_SIGNAL_COLUMN,
    EXIT_SIGNAL_COLUMN,
    OHLCV_COLUMNS,
    SHORT_ENTRY_SIGNAL_COLUMN,
    SHORT_EXIT_SIGNAL_COLUMN,
    SignalModel,
    crossed_above,
    crossed_below,
    require_columns,
)
from time_analysis.models.features import (
    relative_strength_index,
    rolling_zscore,
    safe_percentage_change,
)

EstimatorFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class RandomForestDirectionModel(SignalModel):
    """Walk-forward random forest classifier for future return direction.

    Attributes:
        horizon: number of candles ahead used for the binary direction target
        lookback_window: maximum training rows used for each retraining step
        min_train_size: minimum rows required before the first training step
        retrain_interval: number of candles between model refits
        entry_probability: probability threshold that emits an entry signal
        exit_probability: probability threshold that emits an exit signal
        n_estimators: number of trees in the random forest
        max_depth: maximum tree depth
        min_samples_leaf: minimum samples per leaf
        random_state: deterministic seed used by scikit-learn
    """

    horizon: int = 6
    lookback_window: int = 1500
    min_train_size: int = 300
    retrain_interval: int = 48
    entry_probability: float = 0.56
    exit_probability: float = 0.48
    n_estimators: int = 120
    max_depth: int = 5
    min_samples_leaf: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate random forest walk-forward configuration."""

        _validate_walk_forward_config(
            horizon=self.horizon,
            lookback_window=self.lookback_window,
            min_train_size=self.min_train_size,
            retrain_interval=self.retrain_interval,
            entry_probability=self.entry_probability,
            exit_probability=self.exit_probability,
        )
        if self.n_estimators <= 0:
            msg = "n_estimators must be positive"
            raise ValueError(msg)
        if self.max_depth <= 0:
            msg = "max_depth must be positive"
            raise ValueError(msg)
        if self.min_samples_leaf <= 0:
            msg = "min_samples_leaf must be positive"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required before ML predictions begin.

        :return: minimum training size plus target horizon
        """

        return self.min_train_size + self.horizon

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return walk-forward random forest probabilities and signals.

        :param candles: OHLCV dataframe sorted from oldest to newest candle
        :return: dataframe copy with ML features, probabilities, and signals
        """

        require_columns(candles, OHLCV_COLUMNS)

        def estimator_factory() -> RandomForestClassifier:
            """Create the random forest estimator used for one retraining step.

            :return: configured random forest classifier
            """

            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=self.random_state,
            )

        return _predict_with_walk_forward_classifier(
            candles=candles,
            model_name="random_forest",
            estimator_factory=estimator_factory,
            horizon=self.horizon,
            lookback_window=self.lookback_window,
            min_train_size=self.min_train_size,
            retrain_interval=self.retrain_interval,
            entry_probability=self.entry_probability,
            exit_probability=self.exit_probability,
        )


@dataclass(frozen=True, slots=True)
class HistGradientBoostingDirectionModel(SignalModel):
    """Walk-forward histogram gradient boosting classifier for direction.

    Attributes:
        horizon: number of candles ahead used for the binary direction target
        lookback_window: maximum training rows used for each retraining step
        min_train_size: minimum rows required before the first training step
        retrain_interval: number of candles between model refits
        entry_probability: probability threshold that emits an entry signal
        exit_probability: probability threshold that emits an exit signal
        max_iter: maximum boosting iterations
        learning_rate: shrinkage applied to each tree
        max_leaf_nodes: maximum leaf nodes per tree
        l2_regularization: L2 regularization strength
        random_state: deterministic seed used by scikit-learn
    """

    horizon: int = 6
    lookback_window: int = 1500
    min_train_size: int = 300
    retrain_interval: int = 48
    entry_probability: float = 0.56
    exit_probability: float = 0.48
    max_iter: int = 120
    learning_rate: float = 0.04
    max_leaf_nodes: int = 15
    l2_regularization: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate gradient boosting walk-forward configuration."""

        _validate_walk_forward_config(
            horizon=self.horizon,
            lookback_window=self.lookback_window,
            min_train_size=self.min_train_size,
            retrain_interval=self.retrain_interval,
            entry_probability=self.entry_probability,
            exit_probability=self.exit_probability,
        )
        if self.max_iter <= 0:
            msg = "max_iter must be positive"
            raise ValueError(msg)
        if self.learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.max_leaf_nodes <= 1:
            msg = "max_leaf_nodes must be greater than 1"
            raise ValueError(msg)
        if self.l2_regularization < 0:
            msg = "l2_regularization must be non-negative"
            raise ValueError(msg)

    @property
    def startup_candle_count(self) -> int:
        """Return the candle count required before ML predictions begin.

        :return: minimum training size plus target horizon
        """

        return self.min_train_size + self.horizon

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return walk-forward gradient boosting probabilities and signals.

        :param candles: OHLCV dataframe sorted from oldest to newest candle
        :return: dataframe copy with ML features, probabilities, and signals
        """

        require_columns(candles, OHLCV_COLUMNS)

        def estimator_factory() -> HistGradientBoostingClassifier:
            """Create the gradient boosting estimator for one retraining step.

            :return: configured histogram gradient boosting classifier
            """

            return HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            )

        return _predict_with_walk_forward_classifier(
            candles=candles,
            model_name="hist_gradient_boosting",
            estimator_factory=estimator_factory,
            horizon=self.horizon,
            lookback_window=self.lookback_window,
            min_train_size=self.min_train_size,
            retrain_interval=self.retrain_interval,
            entry_probability=self.entry_probability,
            exit_probability=self.exit_probability,
        )


def _validate_walk_forward_config(
    horizon: int,
    lookback_window: int,
    min_train_size: int,
    retrain_interval: int,
    entry_probability: float,
    exit_probability: float,
) -> None:
    """Validate shared walk-forward model parameters.

    :param horizon: target horizon in candles
    :param lookback_window: maximum rows used for training
    :param min_train_size: minimum rows required for first training
    :param retrain_interval: number of candles between retraining steps
    :param entry_probability: probability threshold for entries
    :param exit_probability: probability threshold for exits
    """

    if horizon <= 0:
        msg = "horizon must be positive"
        raise ValueError(msg)
    if lookback_window <= 10:
        msg = "lookback_window must be greater than 10"
        raise ValueError(msg)
    if min_train_size <= 10:
        msg = "min_train_size must be greater than 10"
        raise ValueError(msg)
    if retrain_interval <= 0:
        msg = "retrain_interval must be positive"
        raise ValueError(msg)
    if not 0.0 < exit_probability < entry_probability < 1.0:
        msg = "probabilities must satisfy 0 < exit < entry < 1"
        raise ValueError(msg)


def _predict_with_walk_forward_classifier(
    candles: pd.DataFrame,
    model_name: str,
    estimator_factory: EstimatorFactory,
    horizon: int,
    lookback_window: int,
    min_train_size: int,
    retrain_interval: int,
    entry_probability: float,
    exit_probability: float,
) -> pd.DataFrame:
    """Run walk-forward training and prediction for a classifier model.

    :param candles: OHLCV dataframe sorted from oldest to newest candle
    :param model_name: prefix used for generated feature and probability columns
    :param estimator_factory: callable that creates a fresh classifier
    :param horizon: target horizon in candles
    :param lookback_window: maximum rows used for each training step
    :param min_train_size: minimum rows required before first training
    :param retrain_interval: number of candles predicted before retraining
    :param entry_probability: probability threshold for entry signals
    :param exit_probability: probability threshold for exit signals
    :return: dataframe copy with ML probabilities and standard signal columns
    """

    result = candles.copy()
    features = _build_direction_features(result)
    future_close = result["close"].shift(-horizon)
    target = (future_close > result["close"]).astype("float64")
    target[future_close.isna()] = np.nan
    probability = pd.Series(np.nan, index=result.index, dtype="float64")

    for prediction_start in range(
        min_train_size + horizon, len(result), retrain_interval
    ):
        train_end = prediction_start - horizon
        train_features = features.iloc[:train_end].dropna()
        train_target = target.reindex(train_features.index).dropna()
        train_features = train_features.reindex(train_target.index)
        if len(train_features) > lookback_window:
            train_features = train_features.tail(lookback_window)
            train_target = train_target.reindex(train_features.index)
        if len(train_features) < min_train_size or train_target.nunique() < 2:
            continue

        estimator = estimator_factory()
        estimator.fit(train_features, train_target.astype("int64"))
        prediction_end = min(prediction_start + retrain_interval, len(result))
        prediction_features = features.iloc[prediction_start:prediction_end].dropna()
        if prediction_features.empty:
            continue

        predicted_probability = _positive_class_probability(
            estimator,
            prediction_features,
        )
        probability.loc[prediction_features.index] = predicted_probability

    probability_column = f"{model_name}_probability"
    result[probability_column] = probability.fillna(0.5)
    for column in features.columns:
        result[f"{model_name}_{column}"] = features[column]
    result[ENTRY_SIGNAL_COLUMN] = crossed_above(
        result[probability_column],
        entry_probability,
    )
    result[EXIT_SIGNAL_COLUMN] = crossed_below(
        result[probability_column],
        exit_probability,
    )
    result[SHORT_ENTRY_SIGNAL_COLUMN] = crossed_below(
        result[probability_column],
        exit_probability,
    )
    result[SHORT_EXIT_SIGNAL_COLUMN] = crossed_above(
        result[probability_column],
        entry_probability,
    )
    return result


def _build_direction_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Build finite numeric features for walk-forward direction classifiers.

    :param candles: OHLCV dataframe
    :return: feature dataframe indexed like the input candles
    """

    close = candles["close"].astype("float64")
    high = candles["high"].astype("float64")
    low = candles["low"].astype("float64")
    volume = candles["volume"].astype("float64")

    features = pd.DataFrame(index=candles.index)
    features["return_1"] = safe_percentage_change(close, 1)
    features["return_3"] = safe_percentage_change(close, 3)
    features["return_12"] = safe_percentage_change(close, 12)
    features["return_48"] = safe_percentage_change(close, 48)
    features["volatility_12"] = features["return_1"].rolling(12, min_periods=12).std()
    features["volatility_48"] = features["return_1"].rolling(48, min_periods=48).std()
    features["rsi_14"] = relative_strength_index(close, 14) / 100.0
    features["volume_z_48"] = rolling_zscore(volume, 48)
    features["range_ratio"] = ((high - low) / close).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    features["close_to_sma_48"] = (
        close / close.rolling(48, min_periods=48).mean()
    ) - 1.0
    return features.replace([np.inf, -np.inf], np.nan)


def _positive_class_probability(
    estimator: object,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return probability of the positive class for a fitted classifier.

    :param estimator: fitted scikit-learn classifier with ``predict_proba``
    :param features: feature dataframe to score
    :return: probability array for class ``1``
    """

    probabilities = estimator.predict_proba(features)
    classes = estimator.classes_
    positive_class_idx = int(np.where(classes == 1)[0][0])
    return probabilities[:, positive_class_idx]
