# backend/providers/alpaca_client.py
import os
import random
import time
from typing import List

import httpx
from httpx import HTTPStatusError

ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

DATA_BASE_URL = "https://data.alpaca.markets"

# simple in-memory cache so synthetic prices are stable-ish per symbol
_price_cache = {}


def _synthetic_price(symbol: str) -> float:
    """Deterministic-ish random walk per symbol so UI has stable demo prices."""
    base = _price_cache.get(symbol.upper())
    if base is None:
        # seed from symbol hash + current day
        seed = hash(symbol.upper()) ^ int(time.time() // 86400)
        rnd = random.Random(seed)
        base = rnd.uniform(20, 300)
    # small random move
    base += random.uniform(-1.0, 1.0)
    _price_cache[symbol.upper()] = base
    return round(base, 2)


def _synthetic_spark(symbol: str, length: int = 120) -> List[float]:
    price = _synthetic_price(symbol)
    spark = []
    for _ in range(length):
        price += random.uniform(-0.8, 0.8)
        spark.append(round(price, 2))
    return spark


def _headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID or "",
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
    }


async def get_last_quote(symbol: str) -> float:
    """
    Get last quote from Alpaca. If keys are missing / unauthorized, fall back to synthetic.
    """
    sym = symbol.upper()
    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        print(f"[alpaca_client] No Alpaca keys configured, using synthetic price for {sym}")
        return _synthetic_price(sym)

    url = f"{DATA_BASE_URL}/v2/stocks/{sym}/trades/latest"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=_headers())
        r.raise_for_status()
        data = r.json()
        price = float(data["trade"]["p"])
        _price_cache[sym] = price
        return price
    except HTTPStatusError as e:
        print(f"[alpaca_client] last_quote HTTP error for {sym}: {e} – using synthetic")
        return _synthetic_price(sym)
    except Exception as e:
        print(f"[alpaca_client] last_quote error for {sym}: {e} – using synthetic")
        return _synthetic_price(sym)


async def get_sparkline(symbol: str, limit: int = 120) -> list[float]:
    """
    Simple sparkline prices. Uses Alpaca bars if available, otherwise synthetic series.
    """
    sym = symbol.upper()

    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        print(f"[alpaca_client] No Alpaca keys configured, using synthetic spark for {sym}")
        return _synthetic_spark(sym, length=limit)

    params = {"timeframe": "1Min", "limit": limit}
    url = f"{DATA_BASE_URL}/v2/stocks/{sym}/bars"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=_headers(), params=params)
        r.raise_for_status()
        data = r.json()
        bars = data.get("bars", [])
        closes = [float(b["c"]) for b in bars] if bars else []
        if not closes:
            return _synthetic_spark(sym, length=limit)
        # ensure we have 'limit' points
        if len(closes) < limit:
            pad = [closes[0]] * (limit - len(closes))
            closes = pad + closes
        return closes[-limit:]
    except HTTPStatusError as e:
        print(f"[alpaca_client] sparkline HTTP error for {sym}: {e} – using synthetic")
        return _synthetic_spark(sym, length=limit)
    except Exception as e:
        print(f"[alpaca_client] sparkline error for {sym}: {e} – using synthetic")
        return _synthetic_spark(sym, length=limit)
