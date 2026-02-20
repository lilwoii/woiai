# backend/training.py

import os
import time
import uuid
import asyncio
from typing import Dict, Any, Optional, Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import open_db
from discord_client import send_discord

# This router is mounted in main.py via:
# from training import router as training_router
# app.include_router(training_router)
router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    backend: Literal["local", "remote"] = "local"
    heavy: bool = False


class TrainingJob(BaseModel):
    id: str
    backend: Literal["local", "remote"]
    heavy: bool
    status: Literal["queued", "running", "completed", "failed"]
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    logs: list[str] = []
    # optional metrics shown in AI Lab
    epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0  # 0..1
    cost_estimate_usd: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_hours: Optional[float] = None


# In-memory job registry (per process)
_jobs: Dict[str, TrainingJob] = {}


def _get_discord_webhook() -> Optional[str]:
    """
    Get Discord webhook:
    - Prefer the DB settings["discord_webhook"]
    - Fall back to DISCORD_WEBHOOK_URL in .env
    """
    try:
        with open_db() as db:
            s = db.get("settings", {})
            webhook = s.get("discord_webhook")
            if webhook:
                return webhook
    except Exception:
        pass
    return os.getenv("DISCORD_WEBHOOK_URL")


async def _log(job_id: str, message: str) -> None:
    """
    Append a log line to the job AND mirror it to Discord (fire-and-forget).
    """
    job = _jobs.get(job_id)
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    if job:
        job.logs.append(line)

    webhook = _get_discord_webhook()
    if webhook:
        try:
            await send_discord(f"[TRAIN {job_id[:8]}] {message}", webhook)
        except Exception:
            # never blow up training because Discord failed
            pass


# ------------------------------------------------------------------
# LOCAL TRAINING (fake but “real” enough to show progress + logs)
# ------------------------------------------------------------------


async def _run_local_training(job_id: str, heavy: bool) -> None:
    """
    Very small "real-ish" local training loop.

    - Trains a toy linear model on synthetic price-like data.
    - Updates job.progress and logs each epoch.
    - If you later install PyTorch / TensorFlow you can swap this out.
    """
    job = _jobs[job_id]
    total_epochs = 20 if heavy else 6
    job.total_epochs = total_epochs

    await _log(job_id, f"Local training started (heavy={heavy}).")

    import random

    # Synthetic dataset: y ≈ 0.1 * x + 1.0 + noise
    xs = [i / 10.0 for i in range(500)]
    ys = [0.1 * x + 1.0 + random.uniform(-0.05, 0.05) for x in xs]

    a, b = 0.0, 0.0  # model params
    lr = 0.05 if heavy else 0.02
    n = len(xs)

    for epoch in range(1, total_epochs + 1):
        grad_a = 0.0
        grad_b = 0.0
        loss = 0.0

        for x, y in zip(xs, ys):
            pred = a * x + b
            err = pred - y
            loss += err * err
            grad_a += 2 * err * x
            grad_b += 2 * err

        loss /= n
        grad_a /= n
        grad_b /= n

        a -= lr * grad_a
        b -= lr * grad_b

        job.epoch = epoch
        job.progress = epoch / total_epochs

        await _log(
            job_id,
            f"[local] epoch {epoch}/{total_epochs} loss={loss:.6f} a={a:.3f} b={b:.3f}",
        )

        # small delay so the UI progress bar actually changes visibly
        await asyncio.sleep(0.3 if heavy else 0.15)

    await _log(job_id, f"Local training finished. Final loss ~{loss:.6f}.")
    job.progress = 1.0


# ------------------------------------------------------------------
# REMOTE TRAINING VIA RUNPOD SERVERLESS
# ------------------------------------------------------------------


async def _run_remote_training(job_id: str, heavy: bool) -> None:
    """
    Remote training via RunPod Serverless.

    We call:
      POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
    then poll:
      GET  https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
    as described in RunPod's docs. :contentReference[oaicite:1]{index=1}
    """
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        raise RuntimeError(
            "RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set in environment."
        )

    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {
        "Authorization": f"{api_key}",  # RunPod expects the raw key in this header
        "Content-Type": "application/json",
    }

    await _log(job_id, "Submitting training job to RunPod /run …")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) Kick off the remote job
        run_resp = await client.post(
            f"{base_url}/run",
            headers=headers,
            json={
                "input": {
                    "job_id": job_id,
                    "heavy": heavy,
                    "project": "wois_assistant",
                    "kind": "trend_training",
                }
            },
        )
        run_resp.raise_for_status()
        run_data = run_resp.json()
        remote_job_id = run_data.get("id")
        status = run_data.get("status")

        if not remote_job_id:
            raise RuntimeError(f"RunPod /run response missing job id: {run_data}")

        await _log(job_id, f"RunPod job id={remote_job_id} status={status}.")

        # 2) Poll /status until COMPLETED / FAILED / etc.
        poll_interval = 5.0
        while True:
            await asyncio.sleep(poll_interval)

            status_resp = await client.get(
                f"{base_url}/status/{remote_job_id}",
                headers=headers,
            )
            status_resp.raise_for_status()
            sdata: Dict[str, Any] = status_resp.json()
            st = sdata.get("status")

            job = _jobs[job_id]

            if st in ("IN_QUEUE", "IN_PROGRESS", "RUNNING"):
                await _log(job_id, f"[remote] Status={st}")
                # OPTIONAL: your RunPod worker can put incremental logs in output.logs
                output = sdata.get("output") or {}
                logs = output.get("logs")
                if isinstance(logs, list):
                    # last few lines only so we don't spam
                    for line in logs[-5:]:
                        await _log(job_id, f"[remote/log] {line}")
                continue

            if st == "COMPLETED":
                output = sdata.get("output") or {}
                await _log(job_id, "[remote] COMPLETED from RunPod.")

                summary = output.get("summary") or output.get("message")
                if summary:
                    await _log(job_id, f"[remote] {summary}")

                cost = output.get("cost_usd")
                gpu_name = output.get("gpu_name")
                gpu_hours = output.get("gpu_hours")

                if cost is not None:
                    job.cost_estimate_usd = float(cost)
                    await _log(job_id, f"Estimated remote cost: ${cost:.4f} USD.")

                if gpu_name:
                    job.gpu_name = str(gpu_name)
                if gpu_hours is not None:
                    job.gpu_hours = float(gpu_hours)

                job.progress = 1.0
                break

            if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                await _log(job_id, f"[remote] job ended with status={st}")
                raise RuntimeError(f"RunPod job ended with status={st}")

            # Safety for any unknown state
            await _log(job_id, f"[remote] Unexpected status={st}, continuing to poll…")


# ------------------------------------------------------------------
# JOB RUNNER + API ENDPOINTS
# ------------------------------------------------------------------


async def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    job.status = "running"
    job.started_at = time.time()

    try:
        if job.backend == "local":
            await _run_local_training(job_id, job.heavy)
        else:
            await _run_remote_training(job_id, job.heavy)
        job.status = "completed"
    except Exception as e:
        job.status = "failed"
        await _log(job_id, f"Training failed: {e}")
    finally:
        job.finished_at = time.time()


@router.post("/start")
async def start_training(req: TrainingStartRequest):
    """
    Kick off a training job.

    The AI Lab UI calls this with:
      POST /training/start
      { "backend": "local" | "remote", "heavy": true/false }

    Returns:
      { "job_id": "...", "backend": "local" }
    """
    job_id = uuid.uuid4().hex
    job = TrainingJob(
        id=job_id,
        backend=req.backend,
        heavy=req.heavy,
        status="queued",
        created_at=time.time(),
    )
    _jobs[job_id] = job

    # Background task
    asyncio.create_task(_run_job(job_id))

    await _log(job_id, f"Queued {req.backend} training (heavy={req.heavy}).")
    return {"job_id": job_id, "backend": req.backend}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Return current status + logs for a job.

    AI Lab polls this to:
      - drive the progress bar (job.progress)
      - show logs (job.logs[])
      - display final cost + GPU info (job.cost_estimate_usd, job.gpu_name)
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return job.dict()
