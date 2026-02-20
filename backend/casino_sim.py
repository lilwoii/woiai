import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _data_path() -> str:
    base = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "casino_interactions.jsonl")


def _append_interaction(record: Dict[str, Any]) -> None:
    try:
        with open(_data_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Never crash the app because of logging.
        pass


def _bj_value(cards: List[int]) -> int:
    # cards are ranks 1..13 (Ace=1, J/Q/K=11/12/13)
    total = 0
    aces = 0
    for r in cards:
        if r == 1:
            aces += 1
            total += 11
        elif r >= 10:
            total += 10
        else:
            total += r
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def _draw_card() -> int:
    # simple uniform rank distribution (not exact deck odds; OK for demo)
    return random.randint(1, 13)


def _bj_basic_action(player_total: int, dealer_up: int, soft: bool) -> str:
    # Very small “good enough” policy for demo.
    # Returns: hit|stand|double
    if player_total <= 11:
        return "hit"
    if player_total == 12:
        return "stand" if 4 <= dealer_up <= 6 else "hit"
    if 13 <= player_total <= 16:
        return "stand" if 2 <= dealer_up <= 6 else "hit"
    if player_total >= 17:
        return "stand"
    return "hit"


def _rank_to_str(r: int) -> str:
    if r == 1:
        return "A"
    if r == 11:
        return "J"
    if r == 12:
        return "Q"
    if r == 13:
        return "K"
    return str(r)


@dataclass
class CasinoStatus:
    running: bool = False
    game: str = "blackjack"
    phase: str = "idle"  # idle|queued|running|paused|stopped|failed
    bankroll: float = 0.0
    unit: float = 10.0
    pnl: float = 0.0
    streak: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    logs: List[Dict[str, Any]] = None
    started_at: Optional[int] = None
    last_tick_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["logs"] = self.logs or []
        return d


class CasinoSimulator:
    def __init__(self, discord_send=None):
        self.status = CasinoStatus(logs=[])
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._discord_send = discord_send
        self._send_to_discord = False
        self._risk = "balanced"

    def _log(self, msg: str, emoji: str = "🎰", level: str = "info"):
        entry = {"id": _now_ms(), "ts": _now_ms(), "emoji": emoji, "level": level, "message": msg}
        self.status.logs = [entry] + (self.status.logs or [])
        self.status.logs = self.status.logs[:200]

    async def start(self, game: str, bankroll: float, unit: float, risk: str, send_to_discord: bool):
        async with self._lock:
            if self.status.running:
                return
            self.status.phase = "queued"
            self.status.game = game
            self.status.bankroll = float(bankroll)
            self.status.unit = max(1.0, float(unit))
            self._risk = risk or "balanced"
            self._send_to_discord = bool(send_to_discord)
            self.status.pnl = 0.0
            self.status.streak = 0
            self.status.wins = 0
            self.status.losses = 0
            self.status.pushes = 0
            self.status.started_at = _now_ms()
            self.status.last_tick_at = None
            self._log(f"Queued {game} sim… bankroll=${bankroll:.2f} unit=${unit:.2f} ({self._risk})", "⏳")

            self.status.running = True
            self.status.phase = "running"
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        async with self._lock:
            self.status.running = False
            if self._task and not self._task.done():
                self._task.cancel()
            self.status.phase = "stopped"
            self._log("Stopped.", "🛑")

    async def _loop(self):
        try:
            while self.status.running:
                self.status.last_tick_at = _now_ms()
                if self.status.game == "roulette":
                    await self._tick_roulette()
                else:
                    await self._tick_blackjack()

                await asyncio.sleep(1.0 if self._risk == "aggressive" else 1.5)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.status.phase = "failed"
            self.status.running = False
            self._log(f"Sim failed: {e}", "❌", "error")

    async def _tick_blackjack(self):
        if self.status.bankroll <= 0:
            self.status.running = False
            self.status.phase = "stopped"
            self._log("Bankroll depleted. Stopping.", "💀")
            return

        bet = min(self.status.unit, self.status.bankroll)

        player = [_draw_card(), _draw_card()]
        dealer = [_draw_card(), _draw_card()]
        dealer_up = dealer[0]

        def is_soft(cards):
            # soft if Ace counted as 11
            total = 0
            aces = 0
            for r in cards:
                if r == 1:
                    aces += 1
                    total += 11
                elif r >= 10:
                    total += 10
                else:
                    total += r
            return aces > 0 and total <= 21

        # Player actions
        while True:
            pt = _bj_value(player)
            if pt >= 21:
                break
            action = _bj_basic_action(pt, 11 if dealer_up == 1 else min(10, dealer_up), is_soft(player))
            if action == "double" and self.status.bankroll >= bet * 2:
                bet *= 2
                player.append(_draw_card())
                break
            if action == "stand":
                break
            player.append(_draw_card())

        pt = _bj_value(player)
        dt = _bj_value(dealer)

        # Dealer hits soft 17? We'll keep simple: hit until 17+
        while dt < 17:
            dealer.append(_draw_card())
            dt = _bj_value(dealer)

        outcome = "push"
        pnl = 0.0
        if pt > 21:
            outcome = "loss"
            pnl = -bet
        elif dt > 21:
            outcome = "win"
            pnl = bet
        elif pt > dt:
            outcome = "win"
            pnl = bet
        elif pt < dt:
            outcome = "loss"
            pnl = -bet

        self.status.bankroll += pnl
        self.status.pnl += pnl

        if outcome == "win":
            self.status.wins += 1
            self.status.streak = max(0, self.status.streak) + 1
            emoji = "✅"
        elif outcome == "loss":
            self.status.losses += 1
            self.status.streak = min(0, self.status.streak) - 1
            emoji = "❌"
        else:
            self.status.pushes += 1
            emoji = "➖"

        msg = (
            f"BJ {emoji} bet ${bet:.2f} | P[{', '.join(_rank_to_str(x) for x in player)}]={pt} "
            f"vs D[{', '.join(_rank_to_str(x) for x in dealer)}]={dt} | bankroll=${self.status.bankroll:.2f}"
        )
        self._log(msg, "🂡")

        _append_interaction(
            {
                "ts": _now_ms(),
                "domain": "casino",
                "game": "blackjack",
                "question": "sim_tick",
                "context": {"player": player, "dealer": dealer, "bet": bet},
                "outcome": {"result": outcome, "pnl": pnl, "bankroll": self.status.bankroll},
            }
        )

        if self._send_to_discord and self._discord_send:
            # send at a low rate
            if random.random() < 0.25:
                await self._discord_send(f"🎰 CasinoSim | {msg}")

    async def _tick_roulette(self):
        if self.status.bankroll <= 0:
            self.status.running = False
            self.status.phase = "stopped"
            self._log("Bankroll depleted. Stopping.", "💀")
            return

        # strategy: cover red/black based on simple momentum of last outcomes
        bet = min(self.status.unit, self.status.bankroll)
        # pick a bet type
        bet_type = random.choice(["red", "black", "even", "odd"]) if self._risk != "conservative" else random.choice(["red", "black"])

        n = random.randint(0, 36)
        red = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
        is_red = n in red
        is_black = n != 0 and not is_red
        is_even = n != 0 and n % 2 == 0
        is_odd = n % 2 == 1

        win = False
        if bet_type == "red":
            win = is_red
        elif bet_type == "black":
            win = is_black
        elif bet_type == "even":
            win = is_even
        elif bet_type == "odd":
            win = is_odd

        pnl = bet if win else -bet
        self.status.bankroll += pnl
        self.status.pnl += pnl

        if win:
            self.status.wins += 1
            self.status.streak = max(0, self.status.streak) + 1
            emoji = "✅"
        else:
            self.status.losses += 1
            self.status.streak = min(0, self.status.streak) - 1
            emoji = "❌"

        color = "🟢" if n == 0 else ("🔴" if is_red else "⚫")
        msg = f"Roulette {emoji} bet {bet_type} ${bet:.2f} | roll={n} {color} | bankroll=${self.status.bankroll:.2f}"
        self._log(msg, "🎯")

        _append_interaction(
            {
                "ts": _now_ms(),
                "domain": "casino",
                "game": "roulette",
                "question": "sim_tick",
                "context": {"bet_type": bet_type, "bet": bet},
                "outcome": {"roll": n, "win": win, "pnl": pnl, "bankroll": self.status.bankroll},
            }
        )

        if self._send_to_discord and self._discord_send:
            if random.random() < 0.25:
                await self._discord_send(f"🎰 CasinoSim | {msg}")

    async def coach(self, game: str, input_text: str) -> Dict[str, Any]:
        # Lightweight rule-based coach.
        game = (game or "").lower()
        t = (input_text or "").strip()
        rec = {"ok": True, "game": game, "input": t, "summary": ""}

        if game == "roulette":
            rec["summary"] = (
                "Roulette is negative-EV long-term. If you still play, keep bets small. "
                "Safer: flat bet red/black or even/odd with strict stop-loss."
            )
            rec["suggestion"] = {"bet": "red", "unit": self.status.unit}
            return rec

        # blackjack: parse very simple formats
        # Example: "P: A,7  D: 9" or "16 vs 10"
        digits = [int(x) for x in t.replace("vs", " ").replace(":", " ").replace(",", " ").split() if x.isdigit()]
        # Try detect "P total" and "dealer up"
        if len(digits) >= 2:
            pt, du = digits[0], digits[1]
            action = _bj_basic_action(pt, du, soft=False)
            friendly = "Hit" if action == "hit" else "Stand" if action == "stand" else "Double"
            rec["summary"] = f"BJ coach: {pt} vs {du} → **{friendly}** (basic-policy demo)."
            rec["suggestion"] = {"action": action}
            return rec

        rec["ok"] = False
        rec["summary"] = (
            "Coach format examples:\n"
            "• Blackjack: '16 vs 10' or 'P: 16 D: 10'\n"
            "• Roulette: anything (advice will be generic)"
        )
        return rec


# ------------------------------
# Public API wrappers (used by main.py)
# ------------------------------

_SIM = CasinoSimulator()


def _as_dict(obj):
    try:
        # dataclass
        from dataclasses import asdict
        return asdict(obj)
    except Exception:
        if hasattr(obj, '__dict__'):
            return dict(obj.__dict__)
        return obj


async def _tick_horse():
    """Simple horse-racing simulator tick (demo only)."""
    global _state
    if not _state["running"]:
        return

    n = random.randint(6, 10)
    odds = [round(random.uniform(2.0, 12.0), 2) for _ in range(n)]
    horses = [{"id": i + 1, "odds": odds[i]} for i in range(n)]

    horses_sorted = sorted(horses, key=lambda h: h["odds"])
    pick = horses_sorted[len(horses_sorted)//2]

    bet = max(1.0, float(_state.get("bet_size", 5.0)))
    bet = min(bet, max(1.0, _state["bank"] * 0.05))

    weights = [1.0 / h["odds"] for h in horses]
    total = sum(weights)
    r = random.random() * total
    cum = 0.0
    winner = horses[0]
    for h, w in zip(horses, weights):
        cum += w
        if r <= cum:
            winner = h
            break

    pnl = bet * (pick["odds"] - 1.0) if winner["id"] == pick["id"] else -bet

    _state["bank"] += pnl
    _state["pnl"] += pnl
    _state["trades"] += 1
    _state["history"].append({
        "ts": time.time(),
        "game": "horse",
        "bet": round(bet, 2),
        "pick": pick["id"],
        "pick_odds": pick["odds"],
        "winner": winner["id"],
        "pnl": round(pnl, 2),
        "bank": round(_state["bank"], 2),
        "race": horses,
    })
    _state["history"] = _state["history"][-200:]

async def casino_start(payload: dict):
    """Start the casino simulator."""
    game = str(payload.get('game','blackjack')).lower()
    bankroll = float(payload.get('bankroll', 500) or 0)
    unit = float(payload.get('unit', 10) or 0)
    risk = str(payload.get('risk','balanced')).lower()
    send_to_discord = bool(payload.get('send_to_discord', False))
    await _SIM.start(game=game, bankroll=bankroll, unit=unit, risk=risk, send_to_discord=send_to_discord)
    return {'ok': True, 'started': True}


async def casino_stop():
    """Stop the casino simulator."""
    return await _SIM.stop()


async def casino_status():
    """Return current simulator status."""
    return _as_dict(_SIM.status)


async def casino_coach(payload: dict):
    """Lightweight coaching helper.

    This does NOT automate any real casino UI. It only provides advice based on
    user-provided state (e.g., blackjack hand).
    """
    game = str(payload.get('game', 'blackjack')).lower()

    def _fmt(summary, detail=None, **extra):
        return {'ok': True, 'game': game, 'summary': summary, 'detail': detail, **extra}


    # --- Blackjack basic strategy (simplified) ---
    if game in ('blackjack', 'bj'):
        # Expect: player = ["A","9"] or "A,9" ; dealer_up = "6"
        player = payload.get('player') or payload.get('player_hand') or ''
        dealer_up = str(payload.get('dealer_up', payload.get('dealer', ''))).strip()
        if isinstance(player, str):
            cards = [c.strip().upper() for c in player.replace('|',',').split(',') if c.strip()]
        else:
            cards = [str(c).strip().upper() for c in (player or [])]

        def card_val(c):
            if c in ('J','Q','K','10'): return 10
            if c == 'A': return 11
            try: return int(c)
            except: return 0

        total = sum(card_val(c) for c in cards)
        aces = sum(1 for c in cards if c == 'A')
        while total > 21 and aces:
            total -= 10
            aces -= 1

        try:
            d = 10 if dealer_up in ('J','Q','K','10') else (11 if dealer_up=='A' else int(dealer_up))
        except:
            d = 0

        # Very simplified guidance
        action = 'STAND'
        reason = ''

        if total <= 11:
            action = 'HIT'
            reason = 'Total is 11 or less.'
        elif total == 12:
            action = 'STAND' if 4 <= d <= 6 else 'HIT'
            reason = 'Stand vs dealer 4-6, otherwise hit.'
        elif 13 <= total <= 16:
            action = 'STAND' if 2 <= d <= 6 else 'HIT'
            reason = 'Stand vs dealer 2-6, otherwise hit.'
        elif total >= 17:
            action = 'STAND'
            reason = 'Total 17+.'

        return {
            'ok': True,
            'game': 'blackjack',
            'player_cards': cards,
            'dealer_up': dealer_up,
            'total': total,
            'summary': action,
            'detail': reason,
            'note': 'Advice only (no automation).'
        }

    # --- Roulette ---
    if game == 'roulette':
        last = payload.get('last_numbers', [])
        if isinstance(last, str):
            last = [x.strip() for x in last.split(',') if x.strip()]
        # simple: diversify with small outside bet
        return _fmt(
            "🎯 Roulette: keep it simple (low risk)",
            "Consider small outside bets (RED/BLACK, EVEN/ODD) + a tiny hedge on 0. Avoid martingale; keep sizing small.",
            last_numbers=last,
        )

    return _fmt(
        "Send more context",
        "Provide the current state (hand/odds/last results) for game-specific advice. Advice only (no automation).",
    )