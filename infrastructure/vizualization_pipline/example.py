from infrastructure.vizualization_pipline import build_visualization, ind, panel

result = build_visualization(
    pairs=["BTC/USDT", "ETH/USDT"],
    timeframe="1h",
    timerange="20250601-20260625",
    indicators=[
        ind("sma_20", "ta.SMA(df, timeperiod=20)", panel="main", color="#f59e0b"),
        ind("rsi_14", "ta.RSI(df, timeperiod=14)", panel="RSI"),
        ind("alpha", "custom.close_zscore(df, window=20)", panel="Alpha"),
    ],
    subplots=[panel("RSI"), panel("Alpha")],
    run_name="my_run",
)

print(result.dashboard_path)
