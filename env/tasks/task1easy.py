from typing import List, Dict, Any
from env.models import Email, Observation, Action, Reward

class Task1Easy:
    TASK_NAME = "Email Classification"
    DIFFICULTY = "easy"
    MAX_STEPS = 10
    DESCRIPTION = "Classify 3 emails into spam, urgent, newsletter, or normal."

    def __init__(self, emails: List[Email]):
        self.emails = emails  # Accept emails arg
        self.state: Dict[str, Any] = {"classified": set()}
        self.done = False
        self.grade = 0.0

    def reset(self) -> Observation:
        self.state = {"classified": set()}
        self.done = False
        self.grade = 0.0
        inbox = self.emails[:3]  # First 3 emails
        return Observation(inbox=inbox, current_task=self.TASK_NAME, ... )  # Full obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        # Implementation for classify grader...
        # (Keep existing grader logic)
        pass
