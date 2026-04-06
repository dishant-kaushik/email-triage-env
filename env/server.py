"""
FastAPI server — exposes OpenEnv HTTP API.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from env.environment import EmailTriageEnv, ALL_TASK_IDS
from env.models import Action, ResetRequest, StepRequest, StepResponse, Observation, Reward

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv-compliant environment for training AI agents on real-world email inbox management tasks.",
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
    return {"status": "healthy", "version": "1.0.0", "environment": "email-triage-env"}

@app.get("/metadata")
async def metadata():
    return {
        "name": "Email Triage OpenEnv",
        "description": "A real-world RL environment where AI agents learn to manage email inboxes.",
        "version": "1.0.0",
        "author": "dishu1999",
        "tags": ["openenv", "email", "triage", "nlp", "reinforcement-learning"],
        "tasks": ALL_TASK_IDS,
    }

@app.get("/schema")
async def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["classify", "prioritize", "label", "reply", "archive", "flag", "skip", "done"]},
                "email_id": {"type": "string", "nullable": True},
                "value": {"type": "string", "nullable": True}
            },
            "required": ["action_type"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "inbox": {"type": "array"},
                "current_task": {"type": "string"},
                "step_count": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "task_info": {"type": "object"},
                "done": {"type": "boolean"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "done": {"type": "boolean"},
                "grade": {"type": "number"},
                "available_tasks": {"type": "array"}
            }
        }
    }

@app.post("/mcp")
async def mcp(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    req_id = body.get("id", 1)
    method = body.get("method", "")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "email-triage-env", "version": "1.0.0"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [{"name": "reset", "description": "Reset the environment", "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}, "seed": {"type": "integer"}}}}, {"name": "step", "description": "Take an action", "inputSchema": {"type": "object", "properties": {"action_type": {"type": "string"}, "email_id": {"type": "string"}, "value": {"type": "string"}}}}, {"name": "state", "description": "Get current state", "inputSchema": {"type": "object", "properties": {}}}]}}
    return {"jsonrpc": "2.0", "id": req_id, "result": {"name": "email-triage-env", "version": "1.0.0", "status": "healthy"}}

@app.get("/tasks")
async def list_tasks():
    tasks = []
    for tid in ALL_TASK_IDS:
        from env.environment import TASK_REGISTRY
        cls = TASK_REGISTRY[tid]
        tasks.append({"task_id": tid, "name": cls.TASK_NAME, "difficulty": cls.DIFFICULTY, "max_steps": cls.MAX_STEPS, "description": cls.DESCRIPTION})
    return {"tasks": tasks}

@app.post("/reset")
async def reset(request: Request):
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
        return StepResponse(observation=obs, reward=reward, done=done, info=info).model_dump()
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
    return {"name": "Email Triage OpenEnv", "docs": "/docs", "health": "/health", "metadata": "/metadata", "schema": "/schema", "mcp": "POST /mcp", "reset": "POST /reset", "step": "POST /step", "state": "GET /state"}
