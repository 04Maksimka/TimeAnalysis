"""Public Bybit OHLCV downloader for research notebooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}
TIMEFRAME_MILLISECONDS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def download_bybit_ohlcv(
    pair: str,
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    category: str = "spot",
    limit: int = 1000,
    sleep_seconds: float = 0.03,
    timeout_seconds: float = 20.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download public Bybit OHLCV candles into a dataframe.

    :param pair: exchange pair such as ``BTC/USDT``
    :param timeframe: timeframe such as ``5m`` or ``1h``
    :param start: inclusive UTC start timestamp
    :param end: inclusive UTC end timestamp
    :param category: Bybit market category, usually ``spot``
    :param limit: maximum candles per request
    :param sleep_seconds: delay between paginated requests
    :param timeout_seconds: HTTP request timeout
    :param session: optional requests session for tests or custom transports
    :return: OHLCV dataframe with ``date``, ``open``, ``high``, ``low``, ``close``,
        and ``volume`` columns
    """

    if timeframe not in BYBIT_INTERVALS:
        available = ", ".join(BYBIT_INTERVALS)
        msg = f"Unsupported timeframe {timeframe!r}. Available: {available}"
        raise ValueError(msg)
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)

    selected_session = session or requests.Session()
    interval_ms = TIMEFRAME_MILLISECONDS[timeframe]
    start_ms = _timestamp_to_milliseconds(start)
    end_ms = _timestamp_to_milliseconds(end)
    symbol = _pair_to_bybit_symbol(pair)

    rows: list[list[str]] = []
    cursor_ms = start_ms
    while cursor_ms <= end_ms:
        request_end_ms = min(
            cursor_ms + (interval_ms * limit) - 1,
            end_ms,
        )
        payload = _request_klines(
            session=selected_session,
            category=category,
            symbol=symbol,
            interval=BYBIT_INTERVALS[timeframe],
            start_ms=cursor_ms,
            end_ms=request_end_ms,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break

        rows.extend(batch)
        max_timestamp_ms = max(int(row[0]) for row in batch)
        next_cursor_ms = max_timestamp_ms + interval_ms
        if next_cursor_ms <= cursor_ms:
            break
        cursor_ms = next_cursor_ms
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return _klines_to_frame(rows)


def save_freqtrade_ohlcv(
    candles: pd.DataFrame,
    freqtrade_dir: Path | str,
    pair: str,
    timeframe: str,
    exchange: str = "bybit",
) -> Path:
    """Save OHLCV candles in the local Freqtrade data directory.

    :param candles: OHLCV dataframe returned by ``download_bybit_ohlcv``
    :param freqtrade_dir: local ``trading/freqtrade`` directory
    :param pair: exchange pair such as ``BTC/USDT``
    :param timeframe: candle timeframe such as ``1h``
    :param exchange: Freqtrade exchange data subdirectory
    :return: path to the saved Feather file
    """

    output_dir = Path(freqtrade_dir) / "user_data" / "data" / exchange
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_slug = pair.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{pair_slug}-{timeframe}.feather"
    candles.reset_index(drop=True).to_feather(output_path)
    return output_path


def download_and_save_bybit_ohlcv(
    pair: str,
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freqtrade_dir: Path | str,
    category: str = "spot",
) -> Path:
    """Download public Bybit candles and save them for the local backtester.

    :param pair: exchange pair such as ``BTC/USDT``
    :param timeframe: timeframe such as ``5m`` or ``1h``
    :param start: inclusive UTC start timestamp
    :param end: inclusive UTC end timestamp
    :param freqtrade_dir: local ``trading/freqtrade`` directory
    :param category: Bybit market category, usually ``spot``
    :return: path to the saved Feather file
    """

    candles = download_bybit_ohlcv(
        pair=pair,
        timeframe=timeframe,
        start=start,
        end=end,
        category=category,
    )
    return save_freqtrade_ohlcv(
        candles=candles,
        freqtrade_dir=freqtrade_dir,
        pair=pair,
        timeframe=timeframe,
    )


def _request_klines(
    session: requests.Session,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Request one page of Bybit kline data.

    :param session: requests session
    :param category: Bybit market category
    :param symbol: Bybit symbol such as ``BTCUSDT``
    :param interval: Bybit interval code
    :param start_ms: page start timestamp in milliseconds
    :param end_ms: page end timestamp in milliseconds
    :param limit: maximum candles per request
    :param timeout_seconds: HTTP request timeout
    :return: decoded Bybit response payload
    """

    response = session.get(
        BYBIT_KLINE_URL,
        params={
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        msg = f"Bybit kline request failed: {payload}"
        raise RuntimeError(msg)
    return payload


def _klines_to_frame(rows: list[list[str]]) -> pd.DataFrame:
    """Convert Bybit kline rows into a normalized OHLCV dataframe.

    :param rows: raw Bybit kline rows
    :return: normalized OHLCV dataframe sorted by date
    """

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    ).drop_duplicates("timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"]),
        unit="ms",
        utc=True,
    )
    return (
        frame[["date", "open", "high", "low", "close", "volume"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def _pair_to_bybit_symbol(pair: str) -> str:
    """Convert a Freqtrade pair into a Bybit spot symbol.

    :param pair: exchange pair such as ``BTC/USDT``
    :return: Bybit symbol such as ``BTCUSDT``
    """

    return pair.replace("/", "").replace(":", "").replace("-", "").upper()


def _timestamp_to_milliseconds(value: str | pd.Timestamp) -> int:
    """Convert a timestamp-like value to UTC epoch milliseconds.

    :param value: timestamp string or pandas timestamp
    :return: UTC epoch milliseconds
    """

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)
