"""
FastAPI server — exposes OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

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

_env = EmailTriageEnv()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "environment": "email-triage-env"}


@app.get("/tasks")
async def list_tasks():
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
async def reset(request: Request):
    """
    Reset the environment. Body is optional.
    Defaults to task_id=classify_emails, seed=42 if no body provided.
    """
    try:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}

        task_id = body.get("task_id", "classify_emails")
        seed = body.get("seed", 42) or 42

        obs = _env.reset(task_id=task_id, seed=seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/step")
async def step(request: StepRequest):
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
    return _env.state()


@app.get("/grade")
async def grade():
    return {"grade": _env.grade(), "task_id": _env._task_id}


@app.get("/")
async def root():
    return {
        "name": "Email Triage OpenEnv",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "GET /state",
    }
