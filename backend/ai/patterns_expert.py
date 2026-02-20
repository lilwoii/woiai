from typing import List, Optional, Dict
from .overlays import swings
import numpy as np

def detect_triangle(ohlc: List[dict]) -> Optional[Dict]:
    closes = [x['c'] for x in ohlc]
    highs, lows = swings(closes, lookback=3)
    if len(highs) < 3 or len(lows) < 3: return None
    hs, ls = highs[-3:], lows[-3:]
    if hs[0][1] > hs[1][1] > hs[2][1] and ls[0][1] < ls[1][1] < ls[2][1]:
        return { 'name': 'Ascending Triangle', 'start': min(hs[0][0], ls[0][0]), 'end': max(hs[2][0], ls[2][0]) }
    if hs[0][1] < hs[1][1] < hs[2][1] and ls[0][1] > ls[1][1] > ls[2][1]:
        return { 'name': 'Descending Triangle', 'start': min(hs[0][0], ls[0][0]), 'end': max(hs[2][0], ls[2][0]) }
    return None

def detect_flag_pennant(ohlc: List[dict]) -> Optional[Dict]:
    closes = [x['c'] for x in ohlc]
    if len(closes) < 40: return None
    run = (closes[-20] - closes[-40]) / max(1e-9, abs(closes[-40]))
    drift = abs(closes[-1]-closes[-20]) / max(1e-9, abs(closes[-20]))
    if abs(run) > 0.05 and drift < 0.02:
        return { 'name': 'Flag/Pennant', 'start': len(closes)-40, 'end': len(closes)-1 }
    return None

def detect_wedge(ohlc: List[dict]) -> Optional[Dict]:
    closes = [x['c'] for x in ohlc]
    highs, lows = swings(closes, lookback=2)
    if len(highs)<3 or len(lows)<3: return None
    hs, ls = highs[-3:], lows[-3:]
    if hs[0][1] < hs[1][1] < hs[2][1] and ls[0][1] < ls[1][1] < ls[2][1]:
        return { 'name': 'Rising Wedge', 'start': min(hs[0][0], ls[0][0]), 'end': max(hs[2][0], ls[2][0]) }
    if hs[0][1] > hs[1][1] > hs[2][1] and ls[0][1] > ls[1][1] > ls[2][1]:
        return { 'name': 'Falling Wedge', 'start': min(hs[0][0], ls[0][0]), 'end': max(hs[2][0], ls[2][0]) }
    return None

def fib_levels(ohlc: List[dict]):
    closes = [x['c'] for x in ohlc]
    window = closes[-60:] if len(closes)>=60 else closes
    lo, hi = min(window), max(window)
    rng = hi - lo if hi>lo else 1e-9
    levels = [0.236, 0.382, 0.5, 0.618]
    return [{ 'name': f'Fib {int(l*100)}%', 'price': hi - l*rng } for l in levels ]

def detect_trend_channel(ohlc: List[dict]) -> Optional[Dict]:
    closes = np.array([x['c'] for x in ohlc[-80:]], dtype=float)
    if len(closes) < 10: return None
    xs = np.arange(len(closes))
    m,b = np.polyfit(xs, closes, 1)
    w = float(np.std(closes - (m*xs + b)))
    if abs(m) < 1e-9: return None
    return { 'name': 'Trend Channel', 'slope': m, 'intercept': b, 'width': w }

def volume_sr(ohlc: List[dict], bins: int = 12):
    prices = [x['c'] for x in ohlc[-300:]]
    vols = [x.get('v',0) for x in ohlc[-300:]]
    if not prices: return []
    lo, hi = min(prices), max(prices)
    step = (hi-lo)/max(1,bins)
    hist = [0]*bins
    for p,v in zip(prices,vols):
        idx = min(bins-1, int((p-lo)/max(1e-9,step)))
        hist[idx]+=v
    zones=[]
    mx = max(hist) if hist else 0
    for i,val in enumerate(hist):
        if val<=0: continue
        zones.append({ 'min': lo+i*step, 'max': lo+(i+1)*step, 'strength': val/mx if mx>0 else 0 })
    return sorted(zones, key=lambda z: z['strength'], reverse=True)[:6]

def candle_annotations(ohlc: List[dict]):
    out=[]
    for i,x in enumerate(ohlc[-120:]):
        o,h,l,c = x['o'],x['h'],x['l'],x['c']
        body = abs(c-o); rng = max(1e-9,h-l)
        upper = h-max(c,o); lower = min(c,o)-l
        idx = len(ohlc)-120+i
        if body/rng < 0.1: out.append({'name':'Doji','idx':idx})
        if c>o and (c-o)/max(1e-9,o) > 0.01 and lower>body*1.5: out.append({'name':'Hammer','idx':idx})
        if o>c and (o-c)/max(1e-9,c) > 0.01 and upper>body*1.5: out.append({'name':'Inverted Hammer','idx':idx})
        if i>0:
            p=ohlc[len(ohlc)-120+i-1]
            if c>o and p['o']>p['c'] and c>p['o'] and o<p['c']:
                out.append({'name':'Bull Engulfing','idx':idx})
            if c<o and p['o']<p['c'] and o>p['c'] and c<p['o']:
                out.append({'name':'Bear Engulfing','idx':idx})
    return out

def build_expert_overlays(ohlc: List[dict]):
    return {
        'triangles': [x for x in [detect_triangle(ohlc)] if x],
        'flags': [x for x in [detect_flag_pennant(ohlc)] if x],
        'wedges': [x for x in [detect_wedge(ohlc)] if x],
        'fib': fib_levels(ohlc),
        'channel': detect_trend_channel(ohlc),
        'vol_zones': volume_sr(ohlc),
        'candles': candle_annotations(ohlc)
    }
