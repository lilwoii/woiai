from pydantic import BaseModel
from typing import Optional, List, Any

class WatchItem(BaseModel):
    symbol: str

class WatchItemInfo(BaseModel):
    symbol: str
    price: float
    change_pct: float
    sparkline: List[float]

class Settings(BaseModel):
    discord_webhook: Optional[str] = None
    use_paper: Optional[bool] = True
    use_crypto: Optional[bool] = False
    signal_schedule_min: Optional[int] = 0
    risk_sl_pct: Optional[float] = 1.5
    risk_tp_pct: Optional[float] = 3.0
    risk_per_trade_pct: Optional[float] = 1.0
    size_mode: Optional[str] = 'risk_pct'
    train_backend: Optional[str] = 'local'  # local | voltagepark
    horizon_min_days: int | None = None
    horizon_max_days: int | None = None
    allow_long_horizon: bool | None = None

class Signal(BaseModel):
    symbol: str
    action: str
    confidence: float
    reasoning: str

class DrawingPayload(BaseModel):
    symbol: str
    objects: list[Any]

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy'|'sell'
    type: str = 'market'  # 'market'|'limit'|'stop'
    qty: Optional[float] = None
    time_in_force: Optional[str] = 'day'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    entry_price: Optional[float] = None  # for SL/TP calc
    sl_pct: Optional[float] = None
    tp_pct: Optional[float] = None
