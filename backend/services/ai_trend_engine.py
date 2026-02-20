from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to import PyTorch for the heavy model; if not installed, we fall back to baseline only.
try:
    import torch
    import torch.nn as nn
except ImportError:  # Torch isn't installed yet
    torch = None
    nn = None

load_dotenv()

SignalType = Literal["BUY", "SELL", "NEUTRAL"]
DirectionType = Literal["UP", "DOWN", "SIDEWAYS"]


class Candle(BaseModel):
    """
    Generic OHLCV candle used by the hybrid trend engine.

    NOTE: This is decoupled from Alpaca/Binance schemas – main.py converts
    provider bars into this type before calling the engine.
    """
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TrendAnalysisResult(BaseModel):
    symbol: str
    trend_id: int
    direction: DirectionType
    baseline_signal: SignalType
    ai_signal: Optional[SignalType]
    final_signal: SignalType
    baseline_confidence: float  # 0–100
    ai_confidence: Optional[float]  # 0–100
    final_confidence: float  # 0–100
    reasoning: str


@dataclass
class TrendFeatures:
    slope: float
    length: int
    volatility: float
    price_change_pct: float
    touches: int


# -------------------------
# Heavy AI model definition
# -------------------------

class HeavyTrendModel(nn.Module if nn is not None else object):
    """
    Simple fully-connected network:
    Input: numeric features extracted from trend
    Output: logits for 3 classes [SELL, NEUTRAL, BUY]

    You can later swap this architecture out or train a bigger one and still
    keep this interface.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_classes: int = 3):
        if nn is None:
            # Torch isn't available; this class will never be instantiated when torch is missing.
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


_heavy_model: Optional[HeavyTrendModel] = None
_heavy_model_available: bool = False

# Default path for the trained heavy model.
# Can be overridden via environment variable HEAVY_TREND_MODEL_PATH.
_DEFAULT_MODEL_PATH = os.getenv(
    "HEAVY_TREND_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "heavy_trend_model.pt"),
)


def _load_heavy_model_if_available() -> None:
    """
    Lazy-loads the heavy trend model if the .pt file exists.
    If torch is not installed or file missing, we just rely on algorithmic logic.
    """
    global _heavy_model, _heavy_model_available

    if _heavy_model is not None or _heavy_model_available:
        return

    if torch is None or nn is None:
        print("[trend_engine] torch not installed, skipping heavy model.")
        _heavy_model_available = False
        return

    model_path = os.path.abspath(_DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"[trend_engine] Heavy model file not found at {model_path}, using baseline only.")
        _heavy_model_available = False
        return

    model = HeavyTrendModel()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _heavy_model = model
    _heavy_model_available = True
    print(f"[trend_engine] Loaded heavy trend model from {model_path}.")


# -------------------------
# Algorithmic trend logic
# -------------------------

def _detect_trend_features(candles: List[Candle]) -> TrendFeatures:
    """
    Extracts simple numeric features from OHLCV candles:
    - Linear regression slope on closes
    - Trend length
    - Volatility
    - Total % change
    - "Touches" on an approximate trendline
    """
    closes = np.array([c.close for c in candles], dtype=float)
    if len(closes) < 5:
        return TrendFeatures(
            slope=0.0,
            length=len(closes),
            volatility=0.0,
            price_change_pct=0.0,
            touches=1,
        )

    x = np.arange(len(closes), dtype=float)
    # simple linear regression slope
    slope, _ = np.polyfit(x, closes, 1)

    price_change_pct = (closes[-1] - closes[0]) / closes[0] * 100.0
    volatility = float(np.std(np.diff(closes))) if len(closes) > 2 else 0.0

    # crude "touch" approximation: how many times price is near the regression line
    line = slope * x + (closes[0] - slope * x[0])
    diff = np.abs(closes - line)
    std_closes = np.std(closes)
    threshold = std_closes * 0.3 if std_closes > 0 else 0.0
    touches = int(np.sum(diff < threshold)) if threshold > 0 else 1

    return TrendFeatures(
        slope=float(slope),
        length=len(closes),
        volatility=volatility,
        price_change_pct=price_change_pct,
        touches=max(touches, 1),
    )


def _baseline_signal_from_features(features: TrendFeatures) -> Tuple[SignalType, float, DirectionType, str]:
    slope = features.slope
    change = features.price_change_pct
    vol = features.volatility
    touches = features.touches

    if abs(slope) < 1e-8:
        direction: DirectionType = "SIDEWAYS"
    else:
        direction = "UP" if slope > 0 else "DOWN"

    # crude confidence from slope + touches + volatility
    slope_strength = min(abs(slope) * 1000, 100.0)  # scaled
    touch_bonus = min(touches * 10, 30.0)
    vol_factor = 10.0 if vol < 0.5 else 0.0  # calmer trend = higher confidence
    baseline_conf = max(15.0, min(slope_strength + touch_bonus + vol_factor, 95.0))

    if direction == "UP" and change > 0:
        sig: SignalType = "BUY"
        reason = "Price is trending up with positive returns and multiple touches along the trendline."
    elif direction == "DOWN" and change < 0:
        sig = "SELL"
        reason = "Price is trending down with negative returns and multiple touches along the trendline."
    else:
        sig = "NEUTRAL"
        reason = "Trend is weak or conflicting; slope and net price change do not strongly align."

    return sig, baseline_conf, direction, reason


# -------------------------
# Heavy AI refinement
# -------------------------

def _ai_refine_signal(features: TrendFeatures) -> Tuple[Optional[SignalType], Optional[float]]:
    """
    Uses the heavy AI model (if available) to refine the signal.

    Returns:
        ai_signal, ai_confidence_pct
        or (None, None) if heavy model is not available.
    """
    _load_heavy_model_if_available()
    if not _heavy_model_available or _heavy_model is None or torch is None:
        return None, None

    # Rough normalization – you can improve this later to match your training preprocessing
    x_vec = np.array(
        [
            features.slope,
            features.length,
            features.volatility,
            features.price_change_pct,
            features.touches,
        ],
        dtype=np.float32,
    )

    x_tensor = torch.from_numpy(x_vec).unsqueeze(0)  # shape [1, 5]
    with torch.no_grad():
        logits = _heavy_model(x_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    class_idx = int(np.argmax(probs))
    prob = float(probs[class_idx]) * 100.0

    # Mapping: 0=SELL, 1=NEUTRAL, 2=BUY
    if class_idx == 0:
        sig: SignalType = "SELL"
    elif class_idx == 2:
        sig = "BUY"
    else:
        sig = "NEUTRAL"

    return sig, prob


def _combine_signals(
    baseline_sig: SignalType,
    baseline_conf: float,
    ai_sig: Optional[SignalType],
    ai_conf: Optional[float],
) -> Tuple[SignalType, float, str]:
    """
    Combines algorithmic and heavy AI decisions.

    Rules:
    - If no AI signal: just use baseline.
    - If AI confidence is noticeably higher: let it override.
    - If they agree: boost confidence.
    - If they conflict and AI isn't much stronger: keep baseline but lower confidence.
    """
    if ai_sig is None or ai_conf is None:
        return baseline_sig, baseline_conf, (
            "Final signal based on algorithmic trend analysis only "
            "(heavy model unavailable or not loaded)."
        )

    if ai_sig == baseline_sig:
        final_sig = baseline_sig
        final_conf = min(100.0, 0.6 * baseline_conf + 0.6 * ai_conf)
        reason = "Algorithmic trend and heavy AI agree on this signal."
        return final_sig, final_conf, reason

    # They disagree
    if ai_conf >= baseline_conf + 5:
        # AI strongly disagrees – give it priority
        final_sig = ai_sig
        final_conf = min(100.0, 0.3 * baseline_conf + 0.9 * ai_conf)
        reason = "Heavy AI model overrides baseline due to stronger confidence."
    else:
        # Baseline holds, but note disagreement
        final_sig = baseline_sig
        final_conf = 0.7 * baseline_conf + 0.3 * ai_conf
        reason = "Algorithmic trend keeps priority, but heavy AI disagreed, reducing overall confidence."

    return final_sig, float(final_conf), reason


# -------------------------
# Public API
# -------------------------

def analyze_trends_for_symbol(symbol: str, candles: List[Candle], trend_id: int = 1) -> TrendAnalysisResult:
    """
    Main entrypoint used by backend.main and the WebSocket loop.

    Steps:
    - Extract numeric features (A: algorithmic)
    - Generate baseline signal (A)
    - Pass features through heavy AI model (B, if available)
    - Combine both into a single final signal + confidence + reasoning
    """
    if not candles:
        return TrendAnalysisResult(
            symbol=symbol,
            trend_id=trend_id,
            direction="SIDEWAYS",
            baseline_signal="NEUTRAL",
            ai_signal=None,
            final_signal="NEUTRAL",
            baseline_confidence=0.0,
            ai_confidence=None,
            final_confidence=0.0,
            reasoning="No candle data provided; unable to analyze trend.",
        )

    features = _detect_trend_features(candles)
    baseline_sig, baseline_conf, direction, base_reason = _baseline_signal_from_features(features)
    ai_sig, ai_conf = _ai_refine_signal(features)
    final_sig, final_conf, combo_reason = _combine_signals(
        baseline_sig, baseline_conf, ai_sig, ai_conf
    )

    full_reason = f"{base_reason} {combo_reason}"
    return TrendAnalysisResult(
        symbol=symbol,
        trend_id=trend_id,
        direction=direction,
        baseline_signal=baseline_sig,
        ai_signal=ai_sig,
        final_signal=final_sig,
        baseline_confidence=round(baseline_conf, 2),
        ai_confidence=round(ai_conf, 2) if ai_conf is not None else None,
        final_confidence=round(final_conf, 2),
        reasoning=full_reason,
    )
