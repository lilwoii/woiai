# backend/services/discord_notifier.py
from __future__ import annotations

import json
from typing import Optional

import requests

from ..config import settings
from .ai_trend_engine import TrendAnalysisResult


def _build_trend_embed(analysis: TrendAnalysisResult) -> dict:
    emoji_map = {
        "BUY": "🟢",
        "SELL": "🔴",
        "NEUTRAL": "⚪",
    }
    direction_emoji = {
        "UP": "📈",
        "DOWN": "📉",
        "SIDEWAYS": "➡️",
    }

    final_emoji = emoji_map.get(analysis.final_signal, "⚪")
    dir_emoji = direction_emoji.get(analysis.direction, "➡️")

    title = f"{final_emoji} {analysis.symbol} – Trend #{analysis.trend_id} ({analysis.final_signal})"
    description = (
        f"{dir_emoji} **Direction:** {analysis.direction}\n"
        f"**Baseline:** {analysis.baseline_signal} ({analysis.baseline_confidence:.1f}%)\n"
        f"**Heavy AI:** {analysis.ai_signal or 'N/A'}"
        f"{f' ({analysis.ai_confidence:.1f}%)' if analysis.ai_confidence is not None else ''}\n"
        f"**Final:** {analysis.final_signal} ({analysis.final_confidence:.1f}%)\n\n"
        f"**Reasoning:** {analysis.reasoning}"
    )

    color = 0x2ecc71 if analysis.final_signal == "BUY" else 0xe74c3c if analysis.final_signal == "SELL" else 0x95a5a6

    return {
        "title": title,
        "description": description,
        "color": color,
    }


def send_trend_analysis_to_discord(
    analysis: TrendAnalysisResult,
    image_path: Optional[str] = None,
) -> None:
    """
    Sends a Discord message for the given trend analysis.
    If image_path is provided, the chart snapshot is sent as an attachment.
    """
    webhook_url = settings.DISCORD_WEBHOOK_URL
    embed = _build_trend_embed(analysis)

    if image_path:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/png")}
            payload = {
                "content": f"📊 AI Trend Analysis for **{analysis.symbol}**",
                "embeds": [embed],
            }
            data = {"payload_json": json.dumps(payload)}
            resp = requests.post(webhook_url, data=data, files=files, timeout=10)
    else:
        payload = {
            "content": f"📊 AI Trend Analysis for **{analysis.symbol}**",
            "embeds": [embed],
        }
        resp = requests.post(webhook_url, json=payload, timeout=10)

    if not resp.ok:
        print(f"[discord_notifier] Failed to send trend analysis: {resp.status_code} {resp.text}")
    else:
        print("[discord_notifier] Trend analysis sent to Discord.")
