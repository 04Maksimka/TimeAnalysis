# Walk-Forward ML Direction Models

This file documents machine-learning models implemented in `ml_direction.py`.
They use the same `SignalModel` interface as deterministic models.

## Target

For each candle, the binary target is:

```text
target_t = 1 if close_{t+horizon} > close_t else 0
```

Rows without a future price are excluded from training. During prediction, each
classifier is trained only on rows whose target would already be known at that
time.

## Features

The current feature set is intentionally small and stable:

- returns over 1, 3, 12, and 48 candles;
- rolling volatility over 12 and 48 candles;
- RSI normalized to `[0, 1]`;
- rolling volume z-score;
- high-low range divided by close;
- close distance from the 48-candle SMA.

## Walk-Forward Protocol

The model repeats this loop:

1. Select only past rows.
2. Use the last `lookback_window` rows.
3. Fit a fresh classifier.
4. Predict probabilities for the next `retrain_interval` candles.
5. Repeat.

This avoids training on the future while still adapting to market regime
changes.

## RandomForestDirectionModel

`RandomForestDirectionModel` uses scikit-learn `RandomForestClassifier`.

Entry:

```text
P(up) crosses above entry_probability
```

Exit:

```text
P(up) crosses below exit_probability
```

The random forest is useful as a nonlinear baseline that can combine several
weak candle features.

## HistGradientBoostingDirectionModel

`HistGradientBoostingDirectionModel` uses scikit-learn
`HistGradientBoostingClassifier`.

Entry and exit rules are the same probability-crossing rules used by the random
forest model.

Gradient boosting is often stronger than a single tree model, but it can
overfit if the lookback window is small or the market regime changes quickly.

## References

- scikit-learn RandomForestClassifier:
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
- scikit-learn HistGradientBoostingClassifier:
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>
- scikit-learn probability calibration and model evaluation:
  <https://scikit-learn.org/stable/modules/calibration.html>
