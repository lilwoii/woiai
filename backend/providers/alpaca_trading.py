import os, httpx

PAPER_BASE = os.getenv('ALPACA_PAPER_BASE_URL', 'https://paper-api.alpaca.markets')
LIVE_BASE = os.getenv('ALPACA_LIVE_BASE_URL', 'https://api.alpaca.markets')

HEADERS = lambda: {
    'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY') or '',
    'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET') or ''
}

def base(use_paper: bool) -> str:
    return PAPER_BASE if use_paper else LIVE_BASE

async def get_account(use_paper: bool):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base(use_paper)}/v2/account", headers=HEADERS())
        r.raise_for_status(); return r.json()

async def get_positions(use_paper: bool):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base(use_paper)}/v2/positions", headers=HEADERS())
        r.raise_for_status(); return r.json()

async def get_orders(use_paper: bool, status='all', limit=200):
    params = {'status': status, 'limit': limit, 'direction': 'desc'}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base(use_paper)}/v2/orders", headers=HEADERS(), params=params)
        r.raise_for_status(); return r.json()

async def place_order(use_paper: bool, symbol: str, qty: int, side: str, type: str='market', time_in_force: str='day', limit_price=None, stop_price=None):
    payload = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': type,
        'time_in_force': time_in_force
    }
    if limit_price is not None: payload['limit_price'] = float(limit_price)
    if stop_price is not None: payload['stop_price'] = float(stop_price)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{base(use_paper)}/v2/orders", headers=HEADERS(), json=payload)
        r.raise_for_status(); return r.json()

async def place_brackets(use_paper: bool, symbol: str, qty: int, sl_price: float, tp_price: float):
    payload = {
        'symbol': symbol,
        'qty': qty,
        'side': 'buy',
        'type': 'market',
        'time_in_force': 'gtc',
        'order_class': 'bracket',
        'take_profit': { 'limit_price': round(float(tp_price), 2) },
        'stop_loss': { 'stop_price': round(float(sl_price), 2) }
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{base(use_paper)}/v2/orders", headers=HEADERS(), json=payload)
        r.raise_for_status(); return r.json()


async def close_position(use_paper: bool, symbol: str):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.delete(f"{base(use_paper)}/v2/positions/{symbol}", headers=HEADERS())
        r.raise_for_status(); return r.json()
