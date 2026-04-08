from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward, Observation, TaskInfo

class Task1Easy:
    TASK_ID = "classify_emails"
    TASK_NAME = "Email Classification"
    DIFFICULTY = "easy"
    MAX_STEPS = 10
    DESCRIPTION = "Classify 3 emails into the correct category: spam, urgent, or newsletter."

    GROUND_TRUTH = {"t1-e001": "spam", "t1-e002": "urgent", "t1-e003": "newsletter"}

    def __init__(self, emails: List[Email], seed: int = 42):
        self.email_list = list(emails)
        self.emails = {e.id: e for e in emails}
        self.results: Dict[str, bool] = {}
        self.step_count = 0
        self.done = False

    def reset(self) -> Observation:
        self.results = {}
        self.step_count = 0
        self.done = False
        return self._obs()

    def step(self, action: Action):
        self.step_count += 1
        if action.action_type == "done":
            self.done = True
            g = self._grade()
            return self._obs(result="Episode ended."), Reward(value=g, cumulative=g, reason="Episode ended.", breakdown={}, penalty=0.0), True
        if action.action_type != "classify":
            g = self._grade()
            return self._obs(error="Use action_type='classify'."), Reward(value=0.001, cumulative=g, reason="Use action_type='classify'.", breakdown={}, penalty=0.0), False
        eid = action.email_id
        if eid not in self.GROUND_TRUTH:
            g = self._grade()
            return self._obs(error=f"Unknown email: {eid}"), Reward(value=0.001, cumulative=g, reason=f"Unknown email: {eid}", breakdown={}, penalty=0.0), False
        if eid in self.results:
            g = self._grade()
            return self._obs(error="Already classified."), Reward(value=0.001, cumulative=g, reason="Already classified.", breakdown={}, penalty=0.0), False
        correct = self.GROUND_TRUTH[eid]
        ok = action.value == correct
        self.results[eid] = ok
        done = len(self.results) == len(self.GROUND_TRUTH)
        self.done = done
        g = self._grade()
        msg = f"Correct! {eid} is {correct}." if ok else f"Wrong. Expected {correct}."
        return self._obs(result=msg), Reward(value=g, cumulative=g, reason=msg, breakdown={"classify": g}, penalty=0.0), done

    def _obs(self, result=None, error=None) -> Observation:
        return Observation(
            inbox=self.email_list,
            current_task=self.TASK_NAME,
            step_count=self.step_count,
            max_steps=self.MAX_STEPS,
            task_info=TaskInfo(
                task_id=self.TASK_ID, task_name=self.TASK_NAME,
                difficulty=self.DIFFICULTY, description=self.DESCRIPTION,
                objectives=[f"Classify {eid} correctly" for eid in self.GROUND_TRUTH],
                hints=["Suspicious sender = spam", "CRITICAL alert = urgent", "Unsubscribe = newsletter"],
                actions_taken=list(self.results.keys()), score_so_far=self._grade(),
            ),
            last_action_result=result, last_action_error=error, done=self.done,
        )

    def _grade(self) -> float:
        if not self.results:
            return 0.001
        raw = sum(1 for v in self.results.values() if v) / len(self.GROUND_TRUTH)
        return max(0.001, min(0.999, raw))

    def grade(self) -> float:
        return self._grade()

    def state(self) -> dict:
        return {"task_id": self.TASK_ID, "step_count": self.step_count, "done": self.done, "grade": self._grade()}
