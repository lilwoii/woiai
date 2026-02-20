from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Zone:
    price_min: float
    price_max: float
    strength: float

@dataclass
class Pattern:
    name: str
    start_idx: int
    end_idx: int

def swings(closes: List[float], lookback: int = 3):
    highs, lows = [], []
    n = len(closes)
    for i in range(lookback, n - lookback):
        p = closes[i]
        if all(p > closes[i-k] for k in range(1, lookback+1)) and all(p > closes[i+k] for k in range(1, lookback+1)):
            highs.append((i, p))
        if all(p < closes[i-k] for k in range(1, lookback+1)) and all(p < closes[i+k] for k in range(1, lookback+1)):
            lows.append((i, p))
    return highs, lows

def zones_from_swings(closes: List[float], tolerance_pct: float = 0.25) -> List[Zone]:
    highs, lows = swings(closes)
    levels = [p for _, p in highs] + [p for _, p in lows]
    levels.sort()
    zones: List[Zone] = []
    for lv in levels:
        matched = False
        for z in zones:
            mid = (z.price_min + z.price_max)/2
            if abs(lv - mid)/max(1e-9, mid)*100 <= tolerance_pct:
                z.price_min = min(z.price_min, lv)
                z.price_max = max(z.price_max, lv)
                z.strength = min(1.0, z.strength + 0.1)
                matched = True
                break
        if not matched:
            zones.append(Zone(price_min=lv*0.998, price_max=lv*1.002, strength=0.3))
    zones = sorted(zones, key=lambda z: z.strength, reverse=True)[:6]
    return zones

def detect_double_top_bottom(closes: List[float]):
    highs, lows = swings(closes)
    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        if abs(p1 - p2)/max(1e-9, (p1+p2)/2)*100 <= 0.4 and i2 - i1 > 3:
            return {'name':'Double Top','start':i1,'end':i2}
    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        if abs(p1 - p2)/max(1e-9, (p1+p2)/2)*100 <= 0.4 and i2 - i1 > 3:
            return {'name':'Double Bottom','start':i1,'end':i2}
    return None

def detect_head_shoulders(closes: List[float]):
    n = len(closes)
    if n < 20: return None
    highs, _ = swings(closes)
    if len(highs) < 3: return None
    i1,p1 = highs[-3]; i2,p2 = highs[-2]; i3,p3 = highs[-1]
    if p2 > p1*1.01 and p2 > p3*1.01 and i3 > i2 > i1:
        return {'name':'Head & Shoulders (tops)','start':i1,'end':i3}
    return None

def build_overlays_from_ohlc(ohlc: List[dict]):
    closes = [x['c'] for x in ohlc]
    zs = zones_from_swings(closes)
    pat = detect_double_top_bottom(closes) or detect_head_shoulders(closes)
    return {
        'zones': [ {'min': z.price_min, 'max': z.price_max, 'strength': z.strength} for z in zs ],
        'pattern': pat
    }
