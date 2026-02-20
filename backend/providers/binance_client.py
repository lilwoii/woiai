import httpx
from typing import List

async def get_last_price(symbol: str) -> float:
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return float(r.json()['price'])

async def get_sparkline(symbol: str, points: int = 120) -> List[float]:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={points}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        kl = r.json()
        return [float(k[4]) for k in kl]
