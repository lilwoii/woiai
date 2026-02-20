# backend/ai/trend_engine.py

from typing import Dict, Any
from .signals import infer_signal
from .ensemble import infer_ensemble
from statistics import mean


def estimate_horizon_days(
    closes,
    tp_pct: float,
    min_days: int,
    max_days: int,
    allow_long: bool,
):
    if len(closes) < 15:
        return min_days, max_days, "unknown"

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append(abs((closes[i] - closes[i - 1]) / closes[i - 1]))

    avg_vol = mean(returns) if returns else 0.0
    if avg_vol <= 0:
        return min_days, max_days, "unknown"

    est_days = (tp_pct / 100.0) / avg_vol
    est_days = max(1, min(120, int(round(est_days))))

    eta_min = max(min_days, est_days - 3)
    eta_max = min(max_days, est_days + 3) if not allow_long else est_days + 5

    if eta_max <= 7:
        label = "weekly"
    elif eta_max <= 30:
        label = "monthly"
    else:
        label = "long"

    return eta_min, eta_max, label


def run_trend_engine(
    symbol: str,
    bars,
    settings: Dict[str, Any],
    heavy: bool = False,
):
    closes = [float(b["c"]) for b in bars if "c" in b]

    action, confidence, reasoning = infer_signal(closes, heavy=heavy)
    ml = infer_ensemble(bars)

    tp_pct = float(settings.get("risk_tp_pct", 3.0))
    eta_min, eta_max, horizon = estimate_horizon_days(
        closes,
        tp_pct,
        int(settings.get("horizon_min_days", 5)),
        int(settings.get("horizon_max_days", 30)),
        bool(settings.get("allow_long_horizon", True)),
    )

    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "ml_class": ml.get("class", "unknown"),
        "ml_prob": max(ml.get("probs", {}).values()) if ml.get("probs") else 0.0,
        "eta_min_days": eta_min,
        "eta_max_days": eta_max,
        "horizon": horizon,
    }
