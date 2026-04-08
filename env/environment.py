<<<<<<< HEAD
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from env.tasks.task1easy import Task1Easy

# Minimal Email model matching existing models.py
class Email(BaseModel):
    id: str
    subject: str
    sender: str
    sender_name: str
    body: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    category: Optional[str] = None

class Observation(BaseModel):
    inbox: list[Email]
    current_task: str
    step_count: int = 0
    max_steps: int = 10
    task_info: dict
    scores_so_far: float = 0.0
    done: bool = False

class Action(BaseModel):
    action_type: str
    email_id: str
    value: str

class Reward(BaseModel):
    value: float
    reason: str

TASK_REGISTRY = {"classifyemails": Task1Easy}
ALL_TASK_IDS = ["classifyemails"]

SAMPLE_EMAILS = [
    Email(id="t1-e001", subject="CONGRATULATIONS! $5M WINNER!", sender="winner@scam.com", sender_name="Prize Center", body="Send bank details...", category="spam"),
    Email(id="t1-e002", subject="CRITICAL: Prod DB Down", sender="alert@company.com", sender_name="PagerDuty", body="Production outage...", category="urgent"),
    Email(id="t1-e003", subject="Weekly Newsletter", sender="news@tech.com", sender_name="Tech Digest", body="Unsubscribe...", category="newsletter")
]

class EmailTriageEnv:
    def __init__(self):
        self.task = None
        self.state = {}
        self.task_id = None

    def reset(self, task_id: str = "classifyemails", seed: int = 42) -> Observation:
        self.task_id = "classifyemails"
        self.task = Task1Easy(emails=SAMPLE_EMAILS)
        obs = self.task.reset()
        self.state = self.task.state
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        return self.task.step(action)
=======
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from env.models import Action, Observation, Reward
from env.tasks.task1_easy import Task1Easy
from env.tasks.task2_medium import Task2Medium
from env.tasks.task3_hard import Task3Hard
from env.data import get_task_emails

TASK_REGISTRY = {
    Task1Easy.TASK_ID: Task1Easy,
    Task2Medium.TASK_ID: Task2Medium,
    Task3Hard.TASK_ID: Task3Hard,
}
ALL_TASK_IDS = list(TASK_REGISTRY.keys())

class EmailTriageEnv:
    def __init__(self):
        self._task = None
        self._task_id = None
        self._seed = 42
        self._episode_rewards = []

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Available: {ALL_TASK_IDS}")
        self._task_id = task_id
        self._seed = seed
        emails = get_task_emails(task_id)
        self._task = TASK_REGISTRY[task_id](emails=emails, seed=seed)
        self._episode_rewards = []
        return self._task.reset()

    def step(self, action: Action):
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        obs, reward, done = self._task.step(action)
        self._episode_rewards.append(reward.value)
        info = {"task_id": self._task_id, "step": obs.step_count,
                "cumulative_reward": getattr(reward, 'cumulative', reward.value),
                "grade": self._task.grade() if done else None}
        return obs, reward, done, info

    def state(self) -> dict:
        if self._task is None:
            return {"status": "not_initialized", "available_tasks": ALL_TASK_IDS}
        return {**self._task.state(), "episode_rewards": self._episode_rewards, "available_tasks": ALL_TASK_IDS}
>>>>>>> b5e7e8b (Fix all tasks: add reset/state/grade, fix Reward fields, fix Observation return)

    @property
    def grade(self) -> float:
<<<<<<< HEAD
        return getattr(self.task, 'grade', 0.0)
=======
        return self._task.grade() if self._task else 0.0

    def close(self):
        self._task = None
        self._task_id = None
        self._episode_rewards = []
>>>>>>> b5e7e8b (Fix all tasks: add reset/state/grade, fix Reward fields, fix Observation return)
