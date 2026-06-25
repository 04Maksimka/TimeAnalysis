from __future__ import annotations

import pandas as pd


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling = series.rolling(window)
    return (series - rolling.mean()) / rolling.std()


def close_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return zscore(df["close"], window=window)
