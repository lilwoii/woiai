# backend/training/remote.py
import os
import asyncio
import time
from typing import Dict, Any

import httpx
from dotenv import load_dotenv

from db import open_db
from discord_client import send_discord

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

# RunPod v2 serverless endpoints: /run (async) and /runsync (blocking) :contentReference[oaicite:0]{index=0}
RUNPOD_RUNSYNC_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync" if RUNPOD_ENDPOINT_ID else None
)


class TrainingError(RuntimeError):
    pass


async def _call_runpod(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fire a synchronous RunPod job and wait for the result.
    This assumes your handler returns {"status": "ok", ...} on success.
    """
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise TrainingError("RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set in env.")

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    req_body = {"input": payload}

    async with httpx.AsyncClient(timeout=1800) as client:  # up to 30 min
        resp = await client.post(RUNPOD_RUNSYNC_URL, headers=headers, json=req_body)
        resp.raise_for_status()
        data = resp.json()

    # RunPod wraps your handler output under "output"
    # https://docs.runpod.io/serverless/overview
    output = data.get("output")
    if output is None:
        raise TrainingError(f"RunPod response missing 'output': {data}")

    return output


# -------- LOCAL TRAINING STUB (replace with your real loop later) --------


async def start_training_local(heavy: bool = False) -> Dict[str, Any]:
    """
    Local training entrypoint. Right now this is a stub that just logs + returns.
    You can later plug a real PyTorch / TF loop here.
    """
    mode = "HEAVY" if heavy else "FAST"
    started_at = time.time()

    # Pull Discord webhook & other settings from DB
    with open_db() as db:
        settings = db.get("settings", {})
        webhook = settings.get("discord_webhook")

    if webhook:
        await send_discord(f"🧠 Starting LOCAL {mode} training job…", webhook)

    # TODO: replace this with your real training code.
    # e.g. call into ai/training_local.py and stream logs to Discord.
    await asyncio.sleep(2.0)

    duration = time.time() - started_at
    summary = {
        "ok": True,
        "backend": "local",
        "mode": mode.lower(),
        "duration_sec": duration,
        "gpu_name": "local-machine",
        "est_cost_usd": 0.0,
        "message": f"Local {mode} training finished in {duration:.1f}s.",
    }

    if webhook:
        await send_discord(
            f"✅ LOCAL {mode} training finished in {duration:.1f}s.", webhook
        )

    return summary


# -------- REMOTE TRAINING VIA RUNPOD --------


async def start_training_remote(heavy: bool = False) -> Dict[str, Any]:
    """
    Kick off training on RunPod and wait for completion.
    Expects handler.py to understand {"mode": "light" | "heavy"} and
    to return {"status": "ok", "gpu_name": "...", "est_cost_usd": 0.123, ...}
    """
    mode = "HEAVY" if heavy else "FAST"
    started_at = time.time()

    with open_db() as db:
        settings = db.get("settings", {})
        webhook = settings.get("discord_webhook")

    if webhook:
        await send_discord(f"🧠 Starting REMOTE {mode} training on RunPod…", webhook)

        # Payload to your handler.py
    payload = {
        "heavy": bool(heavy),                        # <- what handler.py reads
        "mode": "heavy" if heavy else "light",      # optional extra flag
        # later we can add more knobs, e.g. symbol list, epochs, etc.
    }


    try:
        output = await _call_runpod(payload)
    except Exception as e:
        if webhook:
            await send_discord(f"❌ Remote training error: {e}", webhook)
        raise

    duration = time.time() - started_at

    status = output.get("status", "ok")
    gpu_name = output.get("gpu_name", "RunPod GPU")
    est_cost = float(output.get("est_cost_usd", 0.0))

    summary = {
        "ok": status == "ok",
        "backend": "remote",
        "mode": mode.lower(),
        "duration_sec": duration,
        "gpu_name": gpu_name,
        "est_cost_usd": est_cost,
        "raw_output": output,
    }

    if webhook:
        if summary["ok"]:
            await send_discord(
                (
                    f"✅ REMOTE {mode} training done on {gpu_name} in {duration:.1f}s.\n"
                    f"💸 Estimated cost: ~${est_cost:.4f}"
                ),
                webhook,
            )
        else:
            await send_discord(
                f"❌ REMOTE {mode} training failed: {output}", webhook
            )

    return summary
