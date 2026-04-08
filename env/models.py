"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Defines Observation, Action, and Reward models per OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """Represents a single email in the inbox."""
    id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field(..., description="Sender display name")
    body: str = Field(..., description="Email body text")
    timestamp: str = Field(..., description="ISO timestamp of email")
    has_attachment: bool = Field(default=False)
    is_reply: bool = Field(default=False, description="Is this a reply in an existing thread?")
    thread_id: Optional[str] = Field(default=None, description="Thread group ID if part of a thread")

    # Agent-assigned fields (None until agent acts)
    category: Optional[str] = Field(default=None, description="Agent-assigned category")
    priority: Optional[str] = Field(default=None, description="Agent-assigned priority")
    label: Optional[str] = Field(default=None, description="Agent-assigned label")
    reply_draft: Optional[str] = Field(default=None, description="Agent-drafted reply")
    archived: bool = Field(default=False)
    flagged: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    """Task-specific contextual information."""
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    objectives: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    score_so_far: float = Field(default=0.0)
    hints: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    Contains the inbox state and task context the agent uses to decide its next action.
    """
    inbox: List[Email] = Field(..., description="Current inbox emails")
    current_task: str = Field(..., description="Active task identifier")
    step_count: int = Field(..., description="Steps taken so far this episode")
    max_steps: int = Field(..., description="Maximum allowed steps for this task")
    task_info: TaskInfo = Field(..., description="Task objectives and progress info")
    last_action_result: Optional[str] = Field(
        default=None,
        description="Human-readable result of the last action taken"
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if last action was invalid"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

ActionType = Literal["classify", "prioritize", "label", "reply", "archive", "flag", "skip", "done"]

VALID_CATEGORIES = ["spam", "urgent", "normal", "newsletter", "finance", "hr", "tech_support", "social"]
VALID_PRIORITIES = ["high", "medium", "low"]
VALID_LABELS = ["action_required", "fyi", "waiting", "resolved", "duplicate", "archived"]


class Action(BaseModel):
    """
    Agent action on the inbox.

    Examples:
        classify email-001 as 'spam'
        {"action_type": "classify", "email_id": "email-001", "value": "spam"}

        prioritize email-002 as high
        {"action_type": "prioritize", "email_id": "email-002", "value": "high"}

        draft reply to email-003
        {"action_type": "reply", "email_id": "email-003", "value": "Thank you for..."}

        mark episode done
        {"action_type": "done", "email_id": null, "value": null}
    """
    action_type: ActionType = Field(..., description="Type of action to perform")
    email_id: Optional[str] = Field(
        default=None,
        description="Target email ID (not required for 'done' or 'skip')"
    )
    value: Optional[str] = Field(
        default=None,
        description="Action payload: category, priority, label, or reply text"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"action_type": "classify", "email_id": "email-001", "value": "spam"},
                {"action_type": "prioritize", "email_id": "email-002", "value": "high"},
                {"action_type": "label", "email_id": "email-003", "value": "action_required"},
                {"action_type": "reply", "email_id": "email-004", "value": "Hi Sarah, I will get back to you by Friday."},
                {"action_type": "archive", "email_id": "email-005", "value": None},
                {"action_type": "done", "email_id": None, "value": None},
            ]
        }


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Per-step reward signal with breakdown for transparency.
    Provides partial credit signal at every step (not just episode end).
    """
    value: float = Field(..., gt=0.0, lt=1.0, description="Normalized reward for this step (0.0, 1.0)")
    cumulative: float = Field(..., gt=0.0, lt=1.0, description="Cumulative episode reward so far")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective score components"
    )
    reason: str = Field(..., description="Human-readable explanation of this reward")
    penalty: float = Field(default=0.01, description="Penalty applied (e.g. for wasted steps or errors)")


# ---------------------------------------------------------------------------
# API request/response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(..., description="Which task to initialize")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducibility")


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    observation: Observation
    reward_so_far: float
    steps_taken: int
    task_id: str
    is_done: bool
