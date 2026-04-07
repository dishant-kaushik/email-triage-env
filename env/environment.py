from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel
from env.models import Observation, Action, Reward, Email
from env.tasks.task1easy import Task1Easy
from env.tasks.task2medium import Task2Medium
from env.tasks.task3hard import Task3Hard

TASK_REGISTRY = {
    "classifyemails": Task1Easy,
    "prioritizeandlabel": Task2Medium, 
    "fullinboxmanagement": Task3Hard
}

ALL_TASK_IDS = list(TASK_REGISTRY.keys())

# Embedded email data - NO data.py dependency
SAMPLE_EMAILS = [
    Email(id="t1-e001", subject="CONGRATULATIONS! You've won $5,000,000!!!", sender="winner-notify@prize-claims99.com", sender_name="Prize Notification Center", body="Dear Lucky Winner... FIVE MILLION DOLLARS...", category="spam"),
    Email(id="t1-e002", subject="Server down - prod DB unresponsive", sender="alerts@monitoring.company.com", sender_name="PagerDuty Alert", body="CRITICAL ALERT Production database...", category="urgent"),
    Email(id="t1-e003", subject="This week's top articles - TechDigest Newsletter", sender="newsletter@techdigest.io", sender_name="TechDigest Weekly", body="Hi there... Manage preferences", category="newsletter")
]

class EmailTriageEnv:
    def __init__(self):
        self.task = None
        self.state: Dict[str, Any] = {}
        self.task_id = None

    def reset(self, task_id: str = "classifyemails", seed: int = 42) -> Observation:
        self.task_id = task_id
        
        # Load emails based on task (no data.py needed)
        if task_id == "classifyemails":
            emails = SAMPLE_EMAILS[:3]
        elif task_id == "prioritizeandlabel":
            emails = SAMPLE_EMAILS[:5] 
        else:  # fullinboxmanagement
            emails = SAMPLE_EMAILS * 3  # 9 emails
            
        cls = TASK_REGISTRY[task_id]
        self.task = cls(emails=emails)  # Pass emails directly
        obs = self.task.reset()
        self.state = self.task.state
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        obs, reward, done, info = self.task.step(action)
        self.state = self.task.state
        return obs, reward, done, info

    @property
    def grade(self) -> float:
        return getattr(self.task, 'grade', 0.0) if self.task else 0.0
