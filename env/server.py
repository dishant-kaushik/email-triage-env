"""
FastAPI server — exposes OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from env.environment import EmailTriageEnv, ALL_TASK_IDS
from env.models import (
    Action, ResetRequest, StepRequest, StepResponse,
    Observation, Reward
)

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training AI agents on real-world "
        "email inbox management tasks. Supports 3 tasks: easy classification, "
        "medium prioritization, and hard full inbox management."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per HF Space instance)
_env = EmailTriageEnv()


@app.get("/health")
async def health():
    """Health check — returns 200 when server is ready."""
    return {"status": "ok", "version": "1.0.0", "environment": "email-triage-env"}


@app.get("/tasks")
async def list_tasks():
    """List all available task IDs."""
    tasks = []
    for tid in ALL_TASK_IDS:
        from env.environment import TASK_REGISTRY
        cls = TASK_REGISTRY[tid]
        tasks.append({
            "task_id": tid,
            "name": cls.TASK_NAME,
            "difficulty": cls.DIFFICULTY,
            "max_steps": cls.MAX_STEPS,
            "description": cls.DESCRIPTION,
        })
    return {"tasks": tasks}


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Reset the environment for the specified task.
    Returns the initial observation.
    """
    try:
        obs = _env.reset(task_id=request.task_id, seed=request.seed or 42)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Apply an action to the environment.
    Returns observation, reward, done flag, and info dict.
    """
    try:
        obs, reward, done, info = _env.step(request.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        ).model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/state")
async def state():
    """Returns current environment state."""
    return _env.state()


@app.get("/grade")
async def grade():
    """Returns the current episode grade."""
    return {"grade": _env.grade(), "task_id": _env._task_id}


@app.get("/")
async def root():
    """Root endpoint — links to docs."""
    return {
        "name": "Email Triage OpenEnv",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "GET /state",
    }
