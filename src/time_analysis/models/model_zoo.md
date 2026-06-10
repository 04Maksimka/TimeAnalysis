# Default Model Zoo

`default_model_zoo()` returns a comparable set of model instances using the same
`SignalModel` interface:

- `SmaMomentumModel`
- `BuyAndHoldBenchmarkModel`
- `WarmupMomentumHoldModel`
- `EmaTrendModel`
- `DonchianBreakoutModel`
- `BollingerRsiMeanReversionModel`
- `MacdRsiTrendModel`
- `AtrVolatilityBreakoutModel`
- `RandomForestDirectionModel`
- `HistGradientBoostingDirectionModel`

The function accepts:

```python
default_model_zoo(include_ml=True)
```

Set `include_ml=False` when you want a fast first pass over only deterministic
models. Use `include_ml=True` for the full comparison, especially in notebooks
where runtime is less important than research quality.

## Selection Rule

The model zoo does not claim that any model is permanently profitable. The
recommended workflow is:

1. Compare all candidates on the same history.
2. Sort by total return and drawdown-aware metrics.
3. Inspect trades and equity curve.
4. Promote only the best candidate to a Freqtrade adapter.

This makes the “profitable model” a measured result, not a hardcoded promise.
