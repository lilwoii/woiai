# backend/training/__init__.py

import asyncio
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .remote import start_training_local, start_training_remote

# All routes here will be under /training/...
router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    """
    Request body from AILab.jsx:
      { backend: "local" | "remote", heavy: bool }
    """
    backend: str = "local"
    heavy: bool = False


# Simple in-memory job store (fine for your single-process setup)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def _run_training_job(job_id: str) -> None:
    """
    Background task that actually runs local or RunPod training,
    then stores the result back into _jobs[job_id].
    """
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return

    backend = job["backend"]
    heavy = job["heavy"]

    job["status"] = "running"
    job["started_at"] = time.time()
    job["progress"] = 0.0

    try:
        # Call into training/remote.py
        if backend == "local":
            summary = await start_training_local(heavy=heavy)
        else:
            summary = await start_training_remote(heavy=heavy)

        job["status"] = "completed"
        job["finished_at"] = time.time()
        job["progress"] = 1.0
        job["summary"] = summary

        # -------- GPU + cost --------
        # Local summary has gpu_name / est_cost_usd
        gpu_name = summary.get("gpu_name")

        # Remote summary might nest things under raw_output.gpu
        raw = summary.get("raw_output")
        if not gpu_name and isinstance(raw, dict):
            gpu = raw.get("gpu") or {}
            if isinstance(gpu, dict):
                gpu_name = gpu.get("name")

        job["gpu_name"] = gpu_name

        est_cost = (
            summary.get("est_cost_usd")
            or summary.get("est_cost")
            or summary.get("cost")
            or 0.0
        )
        job["cost_estimate_usd"] = float(est_cost)

        # -------- logs --------
        logs = []
        if isinstance(raw, dict) and isinstance(raw.get("logs"), list):
            logs = [str(x) for x in raw["logs"]]
        job["logs"] = logs

        # -------- epochs / progress detail --------
        epochs = summary.get("epochs")
        if isinstance(epochs, int):
            job["epoch"] = epochs
            job["total_epochs"] = epochs

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@router.post("/start")
async def training_start(req: TrainingStartRequest) -> Dict[str, Any]:
    """
    POST /training/start
    Body: { "backend": "local" | "remote", "heavy": bool }

    Returns: { "job_id": "abc123..." }
    """
    backend = req.backend.lower()
    if backend not in ("local", "remote"):
        raise HTTPException(status_code=400, detail="backend must be 'local' or 'remote'")

    job_id = uuid.uuid4().hex[:12]

    job: Dict[str, Any] = {
        "id": job_id,
        "backend": backend,
        "heavy": bool(req.heavy),
        "created_at": time.time(),
        "status": "queued",
        "progress": 0.0,
        "logs": [],
    }

    async with _jobs_lock:
        _jobs[job_id] = job

    # Fire and forget background task
    asyncio.create_task(_run_training_job(job_id))

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def training_status(job_id: str) -> Dict[str, Any]:
    """
    GET /training/status/{job_id}

    This is what AILab.jsx polls.
    """
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    resp: Dict[str, Any] = {
        "job_id": job_id,
        "backend": job["backend"],
        "heavy": job["heavy"],
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "epoch": job.get("epoch"),
        "total_epochs": job.get("total_epochs"),
        "gpu_name": job.get("gpu_name"),
        "cost_estimate_usd": job.get("cost_estimate_usd"),
        "logs": job.get("logs", []),
    }

    if "error" in job:
        resp["error"] = job["error"]

    return resp
