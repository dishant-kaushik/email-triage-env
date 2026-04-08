from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward, Observation, TaskInfo

class Task2Medium:
    TASK_ID = "prioritize_and_label"
    TASK_NAME = "Prioritize and Label"
    DIFFICULTY = "medium"
    MAX_STEPS = 30
    DESCRIPTION = "Set priority, label, and reply for 5 emails."

    GROUND_TRUTH = {
        "t2-e001": {"priority": "high",   "label": "action_required"},
        "t2-e002": {"priority": "low",    "label": "fyi"},
        "t2-e003": {"priority": "high",   "label": "action_required"},
        "t2-e004": {"priority": "low",    "label": "fyi"},
        "t2-e005": {"priority": "medium", "label": "action_required"},
    }
    REPLY_EMAILS = {"t2-e001"}

    def __init__(self, emails: List[Email], seed: int = 42):
        self.email_list = list(emails)
        self.emails = {e.id: e for e in emails}
        self.priorities: Dict[str, bool] = {}
        self.labels: Dict[str, bool] = {}
        self.replies: Dict[str, bool] = {}
        self.step_count = 0
        self.done = False

    def reset(self) -> Observation:
        self.priorities = {}
        self.labels = {}
        self.replies = {}
        self.step_count = 0
        self.done = False
        return self._obs()

    def step(self, action: Action):
        self.step_count += 1
        g = self._grade()
        if action.action_type == "done":
            self.done = True
            return self._obs(result="Episode ended."), Reward(value=g, cumulative=g, reason="Episode ended.", breakdown={}, penalty=0.0), True
        eid = action.email_id
        if not eid or eid not in self.GROUND_TRUTH:
            return self._obs(error=f"Unknown email: {eid}"), Reward(value=0.001, cumulative=g, reason=f"Unknown email: {eid}", breakdown={}, penalty=0.0), False
        gt = self.GROUND_TRUTH[eid]
        if action.action_type == "prioritize":
            if eid in self.priorities:
                return self._obs(error="Already prioritized."), Reward(value=0.001, cumulative=g, reason="Already prioritized.", breakdown={}, penalty=0.0), False
            self.priorities[eid] = action.value == gt["priority"]
            g = self._grade()
            return self._obs(result="Priority set."), Reward(value=g, cumulative=g, reason="Priority set.", breakdown={"priority": g}, penalty=0.0), False
        if action.action_type == "label":
            if eid in self.labels:
                return self._obs(error="Already labeled."), Reward(value=0.001, cumulative=g, reason="Already labeled.", breakdown={}, penalty=0.0), False
            self.labels[eid] = action.value == gt["label"]
            g = self._grade()
            return self._obs(result="Label set."), Reward(value=g, cumulative=g, reason="Label set.", breakdown={"label": g}, penalty=0.0), False
        if action.action_type == "reply":
            if eid not in self.REPLY_EMAILS:
                return self._obs(error="No reply needed."), Reward(value=0.001, cumulative=g, reason="No reply needed.", breakdown={}, penalty=0.0), False
            if eid in self.replies:
                return self._obs(error="Already replied."), Reward(value=0.001, cumulative=g, reason="Already replied.", breakdown={}, penalty=0.0), False
            self.replies[eid] = bool(action.value and len(action.value.strip()) > 20)
            g = self._grade()
            return self._obs(result="Reply recorded."), Reward(value=g, cumulative=g, reason="Reply recorded.", breakdown={"reply": g}, penalty=0.0), False
        return self._obs(error="Unknown action."), Reward(value=0.001, cumulative=g, reason="Unknown action.", breakdown={}, penalty=0.0), False

    def _obs(self, result=None, error=None) -> Observation:
        return Observation(
            inbox=self.email_list,
            current_task=self.TASK_NAME,
            step_count=self.step_count,
            max_steps=self.MAX_STEPS,
            task_info=TaskInfo(
                task_id=self.TASK_ID, task_name=self.TASK_NAME,
                difficulty=self.DIFFICULTY, description=self.DESCRIPTION,
                objectives=["Prioritize all 5", "Label all 5", "Reply to urgent ones"],
                hints=["Invoices are high priority.", "Spam gets low + archived."],
                actions_taken=list(self.priorities.keys()), score_so_far=self._grade(),
            ),
            last_action_result=result, last_action_error=error, done=self.done,
        )

    def _grade(self) -> float:
        total = len(self.GROUND_TRUTH) * 2 + len(self.REPLY_EMAILS)
        scored = (sum(1 for v in self.priorities.values() if v) +
                  sum(1 for v in self.labels.values() if v) +
                  sum(1 for v in self.replies.values() if v))
        return max(0.001, min(0.999, scored / total if total else 0))

    def grade(self) -> float:
        return self._grade()

    def state(self) -> dict:
        return {"task_id": self.TASK_ID, "step_count": self.step_count, "done": self.done, "grade": self._grade()}
