# Financial Math Signal Models

This file documents deterministic rule-based models implemented in
`financial_math.py`. They are not guaranteed to be profitable. They are useful
as known baselines that can be compared on the same data and interface.

## BuyAndHoldBenchmarkModel

`BuyAndHoldBenchmarkModel` is the simplest market benchmark:

```text
enter once near the start of the dataset
hold until the end
```

It is not an alpha model. It answers a necessary research question: did the
candidate strategy outperform simply holding the asset?

## WarmupMomentumHoldModel

`WarmupMomentumHoldModel` is an absolute-momentum filter:

```text
warmup_return = close_{lookback} / close_0 - 1
```

Default configuration:

```text
lookback_candles = 72
minimum_return = 0
require_above_average = False
```

For `1h` candles this means that the model checks the first three days of
market direction before deciding whether to hold the asset.

Entry:

```text
warmup_return > minimum_return
and optional average filter is passed
```

Exit:

```text
no explicit exit; hold until the end of the research interval
```

If the initial trend is weak, the model stays in cash. This makes it a simple
regime filter rather than a constant market exposure benchmark. On the currently
available local `1h` BTC/ETH data from `2024-01-01` to `2026-06-01`, the
default configuration enters BTC and stays out of ETH.

Measured with the vectorized research backtester on those local candles:

```text
BTC/USDT total_return: +72.16%
ETH/USDT total_return:   0.00%
average_return:        +36.08%
worst_drawdown:        -50.09%
```

## EmaTrendModel

`EmaTrendModel` is a trend-following moving-average crossover model.

```text
fast_ema_t = EMA(close, fast_window)
slow_ema_t = EMA(close, slow_window)
```

Entry:

```text
fast_ema crosses above slow_ema
```

Exit:

```text
fast_ema crosses below slow_ema
```

This is a smooth variant of classical moving-average trend following and is
related to the broad time-series momentum literature.

## DonchianBreakoutModel

`DonchianBreakoutModel` uses prior highs and lows:

```text
upper_t = max(high_{t-1}, ..., high_{t-entry_window})
lower_t = min(low_{t-1}, ..., low_{t-exit_window})
```

Entry:

```text
close_t > upper_t
```

Exit:

```text
close_t < lower_t
```

The channel is shifted by one candle so the current candle does not define its
own breakout threshold.

## BollingerRsiMeanReversionModel

`BollingerRsiMeanReversionModel` tries to buy short-term oversold moves.

```text
middle_t = SMA(close, window)
upper_t = middle_t + k * std(close, window)
lower_t = middle_t - k * std(close, window)
```

Entry:

```text
close_t <= lower_t and RSI_t <= oversold_rsi
```

Exit:

```text
close_t >= middle_t or RSI_t >= exit_rsi
```

This model is intentionally different from trend-following candidates. It can
perform better in ranging markets and worse during strong downtrends.

## MacdRsiTrendModel

`MacdRsiTrendModel` combines MACD histogram momentum with RSI filters.

```text
macd_t = EMA(close, fast_window) - EMA(close, slow_window)
signal_t = EMA(macd, signal_window)
histogram_t = macd_t - signal_t
```

Entry:

```text
histogram crosses above 0 and RSI is in the entry range
```

Exit:

```text
histogram crosses below 0 or RSI falls below exit_rsi
```

## AtrVolatilityBreakoutModel

`AtrVolatilityBreakoutModel` uses Average True Range as a volatility-adjusted
breakout threshold:

```text
entry_level_t = close_{t-1} + breakout_multiplier * ATR_{t-1}
exit_level_t = close_{t-1} - exit_multiplier * ATR_{t-1}
```

Entry:

```text
close_t > entry_level_t and close_t > long_ema_t
```

Exit:

```text
close_t < exit_level_t or close_t < long_ema_t
```

## References

- Moskowitz, Ooi, Pedersen, “Time Series Momentum”:
  <https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum>
- John Bollinger, Bollinger Bands overview:
  <https://www.bollingerbands.com/bollinger-bands>
- TA-Lib documentation for RSI, MACD, ATR:
  <https://ta-lib.github.io/ta-doc/>
