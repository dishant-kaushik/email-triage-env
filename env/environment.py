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

    @property
    def grade(self) -> float:
        return getattr(self.task, 'grade', 0.0)
