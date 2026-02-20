import os
import random
from typing import List, Dict, Any

import httpx
from httpx import HTTPStatusError

ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or ""
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET") or ""

BINANCE_URL = os.getenv("BINANCE_REST_URL", "https://api.binance.com")


def _fake_walk(limit: int, start: float = 100.0) -> List[Dict[str, Any]]:
    prices: List[float] = []
    p = float(start)
    for _ in range(limit):
        p += random.uniform(-1.0, 1.0)
        p = max(1.0, p)
        prices.append(p)

    bars: List[Dict[str, Any]] = []
    for i, c in enumerate(prices):
        base = float(c)
        o = base + random.uniform(-0.5, 0.5)
        h = max(o, base) + random.uniform(0, 0.5)
        l = min(o, base) - random.uniform(0, 0.5)
        bars.append(
            {
                "t": i,
                "o": round(o, 2),
                "h": round(h, 2),
                "l": round(l, 2),
                "c": round(base, 2),
                "v": 0,
            }
        )
    return bars


async def alpaca_bars(symbol: str, timeframe: str = "1Min", limit: int = 500) -> List[Dict[str, Any]]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": limit}
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            raw_bars = payload.get("bars", [])

            bars: List[Dict[str, Any]] = []
            for i, b in enumerate(raw_bars):
                bars.append(
                    {
                        "t": i,
                        "o": float(b["o"]),
                        "h": float(b["h"]),
                        "l": float(b["l"]),
                        "c": float(b["c"]),
                        "v": float(b.get("v", 0.0)),
                    }
                )
            if bars:
                return bars
            print(f"[alpaca_bars] No bars for {symbol}, using synthetic.")
            return _fake_walk(limit)
        except HTTPStatusError as e:
            print(f"[alpaca_bars] HTTP error for {symbol}: {e} – using synthetic.")
            return _fake_walk(limit)
        except Exception as e:
            print(f"[alpaca_bars] Unexpected error for {symbol}: {e} – using synthetic.")
            return _fake_walk(limit)


async def binance_klines(symbol: str, interval: str = "1m", limit: int = 500) -> List[Dict[str, Any]]:
    url = f"{BINANCE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()

            bars: List[Dict[str, Any]] = []
            for i, k in enumerate(raw):
                bars.append(
                    {
                        "t": i,
                        "o": float(k[1]),
                        "h": float(k[2]),
                        "l": float(k[3]),
                        "c": float(k[4]),
                        "v": float(k[5]),
                    }
                )
            if bars:
                return bars
            print(f"[binance_klines] No klines for {symbol}, using synthetic.")
            return _fake_walk(limit)
        except HTTPStatusError as e:
            print(f"[binance_klines] HTTP error for {symbol}: {e} – using synthetic.")
            return _fake_walk(limit)
        except Exception as e:
            print(f"[binance_klines] Unexpected error for {symbol}: {e} – using synthetic.")
            return _fake_walk(limit)
