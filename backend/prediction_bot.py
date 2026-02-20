import asyncio
import time
from typing import Any, Dict, List, Optional

from providers.prediction_markets.polymarket_provider import PolymarketProvider
from providers.prediction_markets.kalshi_provider import KalshiProvider

class PredictionBot:
    def __init__(self):
        self.provider_name: str = "polymarket"
        self.provider = PolymarketProvider()
        self.kalshi = KalshiProvider()

        self.running: bool = False
        self.task: Optional[asyncio.Task] = None

        self.config: Dict[str, Any] = {
            "token_id": None,
            "side": "auto",   # auto|yes|no (for binary) - we keep generic
            "max_position_size": 5.0,
            "min_edge": 0.02,
            # Scan cadence. When scanning the whole market, 60s is a sensible default.
            "poll_sec": 60,
            "paper": True,
            "send_to_discord": True,
            # If token_id is not set, scan a batch of markets every poll and hunt tail outcomes
            # (very low or very high implied probability) for later fine-tuning.
            "scan_all_markets": True,
            "scan_limit": 200,
            "tail_threshold": 0.05,  # <=5% or >=95%
        }

        # state
        self.last_prices: List[float] = []
        self.last_decision: Dict[str, Any] = {}

    def set_provider(self, name: str):
        name = (name or "").lower()
        if name not in ("polymarket", "kalshi"):
            raise ValueError("provider must be polymarket or kalshi")
        self.provider_name = name

    def get_provider(self):
        return self.provider if self.provider_name == "polymarket" else self.kalshi

    def decide(self, series: List[float]) -> Dict[str, Any]:
        """Simple shared decision logic (same brain style as stock side).
        This is intentionally lightweight now; you can later swap to the 'true training' model.
        """
        if len(series) < 10:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Not enough data."}

        start = series[0]
        end = series[-1]
        diff = end - start
        # magnitude scaled 0..1 roughly
        mag = abs(diff) / max(1e-6, max(start, 0.01))
        conf = min(1.0, mag * 5.0)

        if diff > 0:
            return {"signal": "BUY", "confidence": conf, "reason": "Rising mid price (probability increasing)."}
        if diff < 0:
            return {"signal": "SELL", "confidence": conf, "reason": "Falling mid price (probability decreasing)."}
        return {"signal": "HOLD", "confidence": 0.2, "reason": "Flat mid price."}

    async def _loop(self, discord_cb=None):
        self.running = True
        while self.running:
            try:
                token_id = self.config.get("token_id")
                prov = self.get_provider()

                # --- Market-wide scan mode (no token_id configured) ---
                if not token_id:
                    if self.provider_name == "polymarket" and self.config.get("scan_all_markets", True):
                        try:
                            limit = int(self.config.get("scan_limit", 200))
                            tail = float(self.config.get("tail_threshold", 0.05))
                            markets = prov.search_markets(q="", limit=limit)
                            candidates = []
                            for m in markets:
                                raw = m.get("raw") or {}
                                title = m.get("title") or raw.get("question") or "(untitled)"
                                # Gamma markets usually expose tokens as a list.
                                tokens = raw.get("tokens") or []
                                for t in tokens:
                                    tid = t.get("token_id") or t.get("tokenId") or t.get("id")
                                    if not tid:
                                        continue
                                    try:
                                        mid = prov.get_mid_price(str(tid))
                                    except Exception:
                                        mid = None
                                    if mid is None:
                                        continue
                                    try:
                                        p = float(mid)
                                    except Exception:
                                        continue
                                    if p <= tail or p >= (1.0 - tail):
                                        candidates.append({
                                            "title": title,
                                            "token_id": str(tid),
                                            "mid_price": p,
                                            "tail": "LOW" if p <= tail else "HIGH",
                                        })
                            candidates = sorted(candidates, key=lambda x: x["mid_price"])[:15]
                            self.last_decision = {
                                "mode": "scan_all",
                                "provider": self.provider_name,
                                "ts": time.time(),
                                "tail_threshold": tail,
                                "candidates": candidates,
                            }
                            if discord_cb and self.config.get("send_to_discord", True) and candidates:
                                lines = []
                                for c in candidates[:10]:
                                    emoji = "🟣" if c["tail"] == "LOW" else "🟢"
                                    lines.append(f"{emoji} {c['mid_price']:.3f}  {c['title'][:70]}  ({c['token_id']})")
                                await discord_cb(
                                    "🎲 PM Scan (tail hunt) — top candidates:\n" + "\n".join(lines)
                                )
                        except Exception as e:
                            if discord_cb and self.config.get("send_to_discord", True):
                                await discord_cb(f"🎲 PM Scan error: {e}")

                    await asyncio.sleep(float(self.config.get("poll_sec", 60)))
                    continue

                # Only polymarket has mid_price in our wrapper right now
                mid = None
                if hasattr(prov, "get_mid_price"):
                    mid = prov.get_mid_price(token_id)
                if mid is None:
                    await asyncio.sleep(float(self.config.get("poll_sec", 60)))
                    continue

                self.last_prices.append(float(mid))
                self.last_prices = self.last_prices[-120:]

                decision = self.decide(self.last_prices[-60:])
                self.last_decision = {
                    **decision,
                    "mid_price": float(mid),
                    "ts": time.time(),
                    "provider": self.provider_name,
                    "token_id": token_id,
                }

                # Optional: place orders (only implemented for polymarket currently)
                if self.provider_name == "polymarket" and decision["confidence"] >= 0.7:
                    # For now we DO NOT auto-place by default unless user sets paper=False AND send_to_discord.
                    if not self.config.get("paper", True):
                        side = "buy" if decision["signal"] == "BUY" else "sell"
                        # price slightly inside the spread (simple)
                        price = float(mid)
                        size = float(self.config.get("max_position_size", 1.0))
                        try:
                            resp = prov.place_limit_order(token_id, side=side, price=price, size=size)
                            if discord_cb and self.config.get("send_to_discord", True):
                                await discord_cb(f"🎲 PM BOT {decision['signal']} ({decision['confidence']:.2f}) @ {price:.3f} size {size}  | {resp}")
                        except Exception as e:
                            if discord_cb and self.config.get("send_to_discord", True):
                                await discord_cb(f"🎲 PM BOT order failed: {e}")

                await asyncio.sleep(float(self.config.get("poll_sec", 60)))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if discord_cb and self.config.get("send_to_discord", True):
                    await discord_cb(f"🎲 PM BOT error: {e}")
                await asyncio.sleep(5.0)
        self.running = False

    def start(self, config: Dict[str, Any], discord_cb=None):
        self.config = {**self.config, **(config or {})}
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self._loop(discord_cb=discord_cb))

    async def stop(self):
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
