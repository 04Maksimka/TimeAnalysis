from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from time_analysis.data_sources.bybit_public import (
    download_bybit_ohlcv,
    save_freqtrade_ohlcv,
)


class FakeResponse:
    """Small fake HTTP response for Bybit downloader tests.

    Attributes:
        payload: decoded JSON payload returned by ``json``
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store a fake response payload.

        :param payload: decoded JSON payload returned by ``json``
        """

        self.payload = payload

    def raise_for_status(self) -> None:
        """Pretend that the HTTP response status is successful."""

    def json(self) -> dict[str, Any]:
        """Return the fake response payload.

        :return: decoded JSON payload
        """

        return self.payload


class FakeSession:
    """Fake requests session that returns predefined kline pages.

    Attributes:
        pages: queued Bybit kline row pages
        calls: request parameter dictionaries received by ``get``
    """

    def __init__(self, pages: list[list[list[str]]]) -> None:
        """Store fake response pages.

        :param pages: ordered list of fake Bybit kline pages
        """

        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        """Return the next fake kline page.

        :param url: requested URL
        :param params: requested query parameters
        :param timeout: request timeout
        :return: fake successful Bybit response
        """

        self.calls.append({"url": url, "params": params, "timeout": timeout})
        page = self.pages.pop(0) if self.pages else []
        return FakeResponse(
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {"list": page},
            }
        )


def test_download_bybit_ohlcv_normalizes_pages() -> None:
    """Check that Bybit kline pages become a sorted OHLCV dataframe."""

    session = FakeSession(
        pages=[
            [
                ["1704067200000", "1", "2", "0.5", "1.5", "10", "15"],
                ["1704070800000", "1.5", "2.5", "1", "2", "20", "40"],
            ],
            [],
        ]
    )

    candles = download_bybit_ohlcv(
        pair="BTC/USDT",
        timeframe="1h",
        start="2024-01-01",
        end="2024-01-01 02:00:00+00:00",
        session=session,
        sleep_seconds=0.0,
    )

    assert list(candles.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(candles) == 2
    assert candles["close"].tolist() == [1.5, 2.0]
    assert session.calls[0]["params"]["symbol"] == "BTCUSDT"


def test_save_freqtrade_ohlcv_writes_feather(tmp_path: Path) -> None:
    """Check that normalized candles are saved to Freqtrade data layout.

    :param tmp_path: pytest temporary directory fixture
    """

    candles = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        }
    )

    path = save_freqtrade_ohlcv(
        candles=candles,
        freqtrade_dir=tmp_path,
        pair="BTC/USDT",
        timeframe="1h",
    )

    assert path == tmp_path / "user_data" / "data" / "bybit" / "BTC_USDT-1h.feather"
    assert path.exists()
    assert len(pd.read_feather(path)) == 2
