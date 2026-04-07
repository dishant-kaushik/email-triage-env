from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward


class Task2Medium:
    TASK_ID = "prioritize_and_label"
    TASK_NAME = "Prioritize and Label"
    DIFFICULTY = "medium"
    MAX_STEPS = 30
    DESCRIPTION = "Set priority, label, and reply for 5 emails."

    GROUND_TRUTH = {
        "t2-e001": {"priority": "high",   "label": "action_required"},
        "t2-e002": {"priority": "low",    "label": "archived"},
        "t2-e003": {"priority": "high",   "label": "action_required"},
        "t2-e004": {"priority": "medium", "label": "fyi"},
        "t2-e005": {"priority": "low",    "label": "archived"},
    }
    REPLY_EMAILS = {"t2-e001", "t2-e003"}

    def __init__(self, emails: List[Email], seed: int = 42):
        self.emails = {e.id: e for e in emails}
        self.priorities: Dict[str, bool] = {}
        self.labels: Dict[str, bool] = {}
        self.replies: Dict[str, bool] = {}

    def step(self, action: Action):
        if action.action_type == "done":
            return Reward(value=self._grade(), reason="Episode ended."), True, {}
        eid = action.email_id
        if eid and eid not in self.GROUND_TRUTH:
            return Reward(value=0.001, cumulative=self._grade(), reason=f"Unknown email: {eid}"), False, {}
        gt = self.GROUND_TRUTH.get(eid, {})
        if action.action_type == "prioritize":
            if eid in self.priorities:
                return Reward(value=0.001, cumulative=self._grade(), reason="Already prioritized."), False, {}
            ok = action.value == gt.get("priority")
            self.priorities[eid] = ok
            return Reward(value=self._grade(), reason="Priority set."), False, {}
        if action.action_type == "label":
            if eid in self.labels:
                return Reward(value=0.001, cumulative=self._grade(), reason="Already labeled."), False, {}
            ok = action.value == gt.get("label")
            self.labels[eid] = ok
            return Reward(value=self._grade(), reason="Label set."), False, {}
        if action.action_type == "reply":
            if eid not in self.REPLY_EMAILS:
                return Reward(value=0.001, cumulative=self._grade(), reason="No reply needed."), False, {}
            if eid in self.replies:
                return Reward(value=0.001, cumulative=self._grade(), reason="Already replied."), False, {}
            ok = bool(action.value and len(action.value.strip()) > 20)
            self.replies[eid] = ok
            return Reward(value=self._grade(), reason="Reply recorded."), False, {}
        return Reward(value=0.001, cumulative=self._grade(), reason="Unknown action."), False, {}

    def _grade(self) -> float:
        total = len(self.GROUND_TRUTH) * 2 + len(self.REPLY_EMAILS)
        scored = (
            sum(1 for v in self.priorities.values() if v) +
            sum(1 for v in self.labels.values() if v) +
            sum(1 for v in self.replies.values() if v)
        )
        raw = scored / total if total else 0
        return max(0.001, min(0.999, raw))

    def get_task_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "task_name": self.TASK_NAME,
            "difficulty": self.DIFFICULTY,
            "description": self.DESCRIPTION,
            "objectives": ["Prioritize all 5", "Label all 5", "Reply to urgent ones"],
            "actions_taken": list(self.priorities.keys()),
            "score_so_far": self._grade(),
            "hints": ["Invoices are high priority.", "Spam gets low + archived."],
        }
