from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel
from env.models import Observation, Action, Reward
from env.data import load_emails
from env.tasks.task1easy import Task1Easy
from env.tasks.task2medium import Task2Medium
from env.tasks.task3hard import Task3Hard

TASK_REGISTRY = {
    "classifyemails": Task1Easy,
    "prioritizeandlabel": Task2Medium,
    "fullinboxmanagement": Task3Hard
}

ALL_TASK_IDS = list(TASK_REGISTRY.keys())

class EmailTriageEnv:
    def __init__(self):
        self.task = None
        self.state: Dict[str, Any] = {}
        self.task_id = None
        self.seed = None

    def reset(self, task_id: str = "classifyemails", seed: int = 42) -> Observation:
        self.seed = seed
        emails = load_emails(task_id, seed)  # Load emails FIRST
        self.task_id = task_id
        cls = TASK_REGISTRY[task_id]
        self.task = cls(emails=emails)  # PASS emails to constructor
        obs = self.task.reset()
        self.state = self.task.state
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        obs, reward, done, info = self.task.step(action)
        self.state = self.task.state
        return obs, reward, done, info

    @property
    def grade(self) -> float:
        return getattr(self.task, 'grade', 0.0) if self.task else 0.0
