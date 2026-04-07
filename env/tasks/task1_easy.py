from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward


class Task1Easy:
    TASK_ID = "classify_emails"
    TASK_NAME = "Email Classification"
    DIFFICULTY = "easy"
    MAX_STEPS = 10
    DESCRIPTION = "Classify 3 emails into the correct category."

    GROUND_TRUTH = {
        "t1-e001": "spam",
        "t1-e002": "urgent",
        "t1-e003": "newsletter",
    }

    def __init__(self, emails: List[Email], seed: int = 42):
        self.emails = {e.id: e for e in emails}
        self.results: Dict[str, bool] = {}

    def step(self, action: Action):
        if action.action_type == "done":
            return Reward(value=self._grade(), reason="Episode ended."), True, {}
        if action.action_type != "classify":
            return Reward(value=0.001, reason="Use classify."), False, {}
        eid = action.email_id
        if eid not in self.GROUND_TRUTH:
            return Reward(value=0.001, reason=f"Unknown email: {eid}"), False, {}
        if eid in self.results:
            return Reward(value=0.001, reason="Already classified."), False, {}
        correct = self.GROUND_TRUTH[eid]
        ok = action.value == correct
        self.results[eid] = ok
        done = len(self.results) == len(self.GROUND_TRUTH)
        msg = f"Correct! {eid} is {correct}." if ok else f"Wrong. Expected {correct}."
        return Reward(value=self._grade(), reason=msg), done, {}

    def _grade(self) -> float:
        if not self.results:
            return 0.001
        raw = sum(1 for v in self.results.values() if v) / len(self.GROUND_TRUTH)
        return max(0.001, min(0.999, raw))

    def get_task_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "task_name": self.TASK_NAME,
            "difficulty": self.DIFFICULTY,
            "description": self.DESCRIPTION,
            "objectives": [f"Classify {e}" for e in self.GROUND_TRUTH],
            "actions_taken": list(self.results.keys()),
            "score_so_far": self._grade(),
            "hints": [
                "Prize claim emails are spam.",
                "CRITICAL alerts are urgent.",
                "Unsubscribe links mean newsletter.",
            ],
        }
