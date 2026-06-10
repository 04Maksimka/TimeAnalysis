# Shared SignalModel Interface

All research models in `time_analysis.models` use the same contract:

```python
class SignalModel:
    @property
    def startup_candle_count(self) -> int: ...

    def predict(self, candles: pd.DataFrame) -> pd.DataFrame: ...
```

The input is an OHLCV dataframe sorted from oldest to newest candle. The output
is a copy of the input with any model-specific feature columns and two standard
boolean columns:

- `long_entry_signal`
- `long_exit_signal`
- `short_entry_signal` (optional)
- `short_exit_signal` (optional)

The model must not mutate the input dataframe. This makes the same model usable
from notebooks, lightweight vectorized backtests, and Freqtrade adapters.

## Design Rules

- Keep models exchange-independent.
- Avoid future leakage: indicators should use only current and past candles.
- Use event-like signals when possible, such as crossovers or breakouts.
- Keep runtime adapters thin; adapters should map model signals to runtime
  columns such as Freqtrade `enter_long` and `exit_long`.

## Evaluation

The vectorized research evaluator in `time_analysis.backtesting` applies a
one-candle signal delay:

```text
signal at t -> exposure for return from t to t + 1
```

This is intentionally conservative compared with using the current candle's
return immediately after the signal is calculated.
