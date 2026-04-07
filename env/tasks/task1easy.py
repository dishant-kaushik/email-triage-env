from typing import List, Dict, Any
from env.models import Email, Observation, Action, Reward

class Task1Easy:
    TASK_NAME = "Email Classification"
    DIFFICULTY = "easy" 
    MAX_STEPS = 10
    DESCRIPTION = "Classify 3 emails into categories."

    def __init__(self, emails: List[Email]):
        self.emails = emails[:3]  # Take first 3
        self.state = {"classified": {}}
        self.done = False
        self.grade = 0.0
        self.step_count = 0

    def reset(self) -> Observation:
        self.state = {"classified": {}}
        self.done = False
        self.grade = 0.0
        self.step_count = 0
        return Observation(
            inbox=self.emails,
            current_task=self.TASK_NAME,
            step_count=0,
            max_steps=self.MAX_STEPS,
            task_info={"task_id": "classifyemails", "task_name": self.TASK_NAME},
            scores_so_far=0.0,
            done=False
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.step_count += 1
        email_id = action.email_id
        category = action.value
        
        # Simple grader
        ground_truth = {
            "t1-e001": "spam",
            "t1-e002": "urgent", 
            "t1-e003": "newsletter"
        }
        
        correct = ground_truth.get(email_id) == category
        self.state["classified"][email_id] = category
        self.grade = len([e for e, c in self.state["classified"].items() 
                         if ground_truth.get(e) == c]) / 3.0
        
        reward = Reward(value=0.333 if correct else 0.0, reason="Correct!" if correct else "Wrong category")
        
        done = len(self.state["classified"]) >= 3 or self.step_count >= self.MAX_STEPS
        
        obs = Observation(
            inbox=self.emails,
            current_task=self.TASK_NAME,
            step_count=self.step_count,
            max_steps=self.MAX_STEPS,
            scores_so_far=self.grade,
            done=done
        )
        
        return obs, reward, done, {"info": "classified"}
