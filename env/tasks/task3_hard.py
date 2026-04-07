from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward


class Task3Hard:
    TASK_ID = "full_inbox_management"
    TASK_NAME = "Full Inbox Management"
    DIFFICULTY = "hard"
    MAX_STEPS = 60
    DESCRIPTION = "Manage 10 emails: classify, prioritize, label, flag, reply, archive."

    GROUND_TRUTH = {
        "t3-e001": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
        "t3-e002": {"category": "spam",       "priority": "low",    "label": "archived",        "flag": False, "reply": False, "archive": True},
        "t3-e003": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
        "t3-e004": {"category": "newsletter", "priority": "low",    "label": "archived",        "flag": False, "reply": False, "archive": True},
        "t3-e005": {"category": "normal",     "priority": "medium", "label": "fyi",             "flag": False, "reply": False, "archive": False},
        "t3-e006": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
        "t3-e007": {"category": "spam",       "priority": "low",    "label": "archived",        "flag": False, "reply": False, "archive": True},
        "t3-e008": {"category": "normal",     "priority": "medium", "label": "fyi",             "flag": False, "reply": False, "archive": False},
        "t3-e009": {"category": "newsletter", "priority": "low",    "label": "archived",        "flag": False, "reply": False, "archive": True},
        "t3-e010": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
    }

    def __init__(self, emails: List[Email]):
        self.emails = {e.id: e for e in emails}
        self.categories: Dict[str, bool] = {}
        self.priorities: Dict[str, bool] = {}
        self.labels: Dict[str, bool] = {}
        self.flags: Dict[str, bool] = {}
        self.replies: Dict[str, bool] = {}
        self.archives: Dict[str, bool] = {}

    def step(self, action: Action):
        if action.action_type == "done":
            return Reward(value=self._grade(), reason="Episode ended."), True, {}
        eid = action.email_id
        if eid and eid not in self.GROUND_TRUTH:
            return Reward(value=0.001, reason=f"Unknown email: {eid}"), False, {}
        gt = self.GROUND_TRUTH.get(eid, {})
        if action.action_type == "classify":
            if eid in self.categories:
                return Reward(value=0.001, reason="Already classified."), False, {}
            ok = action.value == gt.get("category")
            self.categories[eid] = ok
            return Reward(value=self._grade(), reason="Classified."), False, {}
        if action.action_type == "prioritize":
            if eid in self.priorities:
                return Reward(value=0.001, reason="Already prioritized."), False, {}
            ok = action.value == gt.get("priority")
            self.priorities[eid] = ok
            return Reward(value=self._grade(), reason="Priority set."), False, {}
        if action.action_type == "label":
            if eid in self.labels:
                return Reward(value=0.001, reason="Already labeled."), False, {}
            ok = action.value == gt.get("label")
            self.labels[eid] = ok
            return Reward(value=self._grade(), reason="Labeled."), False, {}
        if action.action_type == "flag":
            if eid in self.flags:
                return Reward(value=0.001, reason="Already flagged."), False, {}
            ok = gt.get("flag", False)
            self.flags[eid] = ok
            return Reward(value=self._grade(), reason="Flagged."), False, {}
        if action.action_type == "reply":
            if not gt.get("reply", False):
                return Reward(value=0.001, reason="No reply needed."), False, {}
            if eid in self.replies:
                return Reward(value=0.001, reason="Already replied."), False, {}
            ok = bool(action.value and len(action.value.strip()) > 20)
            self.replies[eid] = ok
            return Reward(value=self._grade(), reason="Reply recorded."), False, {}
        if action.action_type == "archive":
            if eid in self.archives:
                return Reward(value=0.001, reason="Already archived."), False, {}
            ok = gt.get("archive", False)
            self.archives[eid] = ok
            return Reward(value=self._grade(), reason="Archived."), False, {}
        if action.action_type == "skip":
            return Reward(value=0.001, reason="Skipped."), False, {}
        return Reward(value=0.001, reason="Unknown action."), False, {}

    def _grade(self) -> float:
        urgent = [k for k, v in self.GROUND_TRUTH.items() if v["flag"]]
        archive = [k for k, v in self.GROUND_TRUTH.items() if v["archive"]]
        reply = [k for k, v in self.GROUND_TRUTH.items() if v["reply"]]
        total = len(self.GROUND_TRUTH) * 3 + len(urgent) + len(reply) + len(archive)
        scored = (
            sum(1 for v in self.categories.values() if v) +
            sum(1 for v in self.priorities.values() if v) +
            sum(1 for v in self.labels.values() if v) +
            sum(1 for v in self.flags.values() if v) +
            sum(1 for v in self.replies.values() if v) +
            sum(1 for v in self.archives.values() if v)
        )
        raw = scored / total if total else 0
        return max(0.001, min(0.999, raw))

    def get_task_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "task_name": self.TASK_NAME,
            "difficulty": self.DIFFICULTY,
            "description": self.DESCRIPTION,
            "objectives": ["Classify all 10", "Prioritize all 10", "Label all 10", "Flag urgent", "Reply to urgent", "Archive spam/newsletter"],
            "actions_taken": list(self.categories.keys()),
            "score_so_far": self._grade(),
            "hints": ["Classify first.", "Urgent needs flag+reply.", "Spam/newsletter gets archived."],
        }
