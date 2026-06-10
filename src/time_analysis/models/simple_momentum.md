# SmaMomentumModel Math

`SmaMomentumModel` is a deterministic long-only baseline model based on a fast
and a slow simple moving average. It converts candle close prices into
indicator columns and boolean entry/exit signal columns.

## Input

The model expects a pandas dataframe with at least one column:

- `close`: candle close price at time index `t`

Let the close price at time `t` be:

```text
C_t
```

The model has two window parameters:

```text
fast_window = F
slow_window = S
```

The required constraint is:

```text
0 < F < S
```

## Simple Moving Averages

The fast simple moving average at time `t` is:

```text
fast_sma_t = (C_t + C_{t-1} + ... + C_{t-F+1}) / F
```

The slow simple moving average at time `t` is:

```text
slow_sma_t = (C_t + C_{t-1} + ... + C_{t-S+1}) / S
```

The model does not emit stable indicators until enough candles exist for each
window. For this reason, the strategy startup candle count is equal to:

```text
startup_candle_count = S
```

## Momentum Score

The model also computes a normalized relative distance between the fast and slow
averages:

```text
sma_momentum_score_t = (fast_sma_t / slow_sma_t) - 1
```

Interpretation:

- `sma_momentum_score_t > 0`: the fast average is above the slow average.
- `sma_momentum_score_t = 0`: both averages are equal.
- `sma_momentum_score_t < 0`: the fast average is below the slow average.

Missing values before the slow window is available are filled with `0.0`.

## Entry Signal

A long entry signal is produced only on the candle where the fast average crosses
from below the slow average to above it:

```text
long_entry_signal_t =
  fast_sma_t > slow_sma_t
  and fast_sma_{t-1} <= slow_sma_{t-1}
```

This makes the signal event-based. The model does not mark every candle in an
uptrend as an entry signal. It marks only the crossover candle.

## Exit Signal

A long exit signal is produced only on the candle where the fast average crosses
from above the slow average to below it:

```text
long_exit_signal_t =
  fast_sma_t < slow_sma_t
  and fast_sma_{t-1} >= slow_sma_{t-1}
```

This is also event-based. The model exits when the short-term average loses its
lead over the long-term average.

## Output Columns

The returned dataframe is a copy of the input with these additional columns:

- `sma_fast`
- `sma_slow`
- `sma_momentum_score`
- `long_entry_signal`
- `long_exit_signal`

The input dataframe is not mutated.

## Trading Meaning

The model assumes that a fast average crossing above a slow average can indicate
positive short-term momentum, and that the reverse crossover can indicate that
the momentum has weakened.

This is a baseline model, not a production trading edge. Its main purpose is to
provide a simple, testable contract between research code and the Freqtrade
runtime adapter.
