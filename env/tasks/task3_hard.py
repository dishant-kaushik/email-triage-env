from __future__ import annotations
from typing import Any, Dict, List
from env.models import Email, Action, Reward, Observation, TaskInfo

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
        "t3-e004": {"category": "normal",     "priority": "medium", "label": "fyi",             "flag": False, "reply": False, "archive": False},
        "t3-e005": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
        "t3-e006": {"category": "normal",     "priority": "low",    "label": "resolved",        "flag": False, "reply": False, "archive": True},
        "t3-e007": {"category": "normal",     "priority": "low",    "label": "fyi",             "flag": False, "reply": False, "archive": True},
        "t3-e008": {"category": "normal",     "priority": "medium", "label": "action_required", "flag": False, "reply": False, "archive": False},
        "t3-e009": {"category": "newsletter", "priority": "low",    "label": "archived",        "flag": False, "reply": False, "archive": True},
        "t3-e010": {"category": "urgent",     "priority": "high",   "label": "action_required", "flag": True,  "reply": True,  "archive": False},
    }

    def __init__(self, emails: List[Email], seed: int = 42):
        self.email_list = list(emails)
        self.emails = {e.id: e for e in emails}
        self.categories: Dict[str, bool] = {}
        self.priorities: Dict[str, bool] = {}
        self.labels: Dict[str, bool] = {}
        self.flags: Dict[str, bool] = {}
        self.replies: Dict[str, bool] = {}
        self.archives: Dict[str, bool] = {}
        self.step_count = 0
        self.done = False

    def reset(self) -> Observation:
        self.categories = {}
        self.priorities = {}
        self.labels = {}
        self.flags = {}
        self.replies = {}
        self.archives = {}
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
            return self._obs(error=f"Unknown: {eid}"), Reward(value=0.001, cumulative=g, reason=f"Unknown: {eid}", breakdown={}, penalty=0.0), False
        gt = self.GROUND_TRUTH[eid]
        if action.action_type == "classify":
            if eid in self.categories:
                return self._obs(error="Already classified."), Reward(value=0.001, cumulative=g, reason="Already classified.", breakdown={}, penalty=0.0), False
            self.categories[eid] = action.value == gt["category"]
            g = self._grade()
            return self._obs(result="Classified."), Reward(value=g, cumulative=g, reason="Classified.", breakdown={"classify": g}, penalty=0.0), False
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
            return self._obs(result="Labeled."), Reward(value=g, cumulative=g, reason="Labeled.", breakdown={"label": g}, penalty=0.0), False
        if action.action_type == "flag":
            if eid in self.flags:
                return self._obs(error="Already flagged."), Reward(value=0.001, cumulative=g, reason="Already flagged.", breakdown={}, penalty=0.0), False
            self.flags[eid] = gt.get("flag", False)
            g = self._grade()
            return self._obs(result="Flagged."), Reward(value=g, cumulative=g, reason="Flagged.", breakdown={"flag": g}, penalty=0.0), False
        if action.action_type == "reply":
            if not gt.get("reply", False):
                return self._obs(error="No reply needed."), Reward(value=0.001, cumulative=g, reason="No reply needed.", breakdown={}, penalty=0.0), False
            if eid in self.replies:
                return self._obs(error="Already replied."), Reward(value=0.001, cumulative=g, reason="Already replied.", breakdown={}, penalty=0.0), False
            self.replies[eid] = bool(action.value and len(action.value.strip()) > 20)
            g = self._grade()
            return self._obs(result="Reply recorded."), Reward(value=g, cumulative=g, reason="Reply recorded.", breakdown={"reply": g}, penalty=0.0), False
        if action.action_type == "archive":
            if eid in self.archives:
                return self._obs(error="Already archived."), Reward(value=0.001, cumulative=g, reason="Already archived.", breakdown={}, penalty=0.0), False
            self.archives[eid] = gt.get("archive", False)
            g = self._grade()
            return self._obs(result="Archived."), Reward(value=g, cumulative=g, reason="Archived.", breakdown={"archive": g}, penalty=0.0), False
        if action.action_type == "skip":
            return self._obs(), Reward(value=0.001, cumulative=g, reason="Skipped.", breakdown={}, penalty=0.0), False
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
                objectives=["Classify all 10", "Prioritize all 10", "Label all 10", "Flag urgent", "Reply to urgent", "Archive spam/newsletter"],
                hints=["Classify first.", "Urgent needs flag+reply.", "Spam/newsletter gets archived."],
                actions_taken=list(self.categories.keys()), score_so_far=self._grade(),
            ),
            last_action_result=result, last_action_error=error, done=self.done,
        )

    def _grade(self) -> float:
        flag_needed = [k for k, v in self.GROUND_TRUTH.items() if v["flag"]]
        reply_needed = [k for k, v in self.GROUND_TRUTH.items() if v["reply"]]
        archive_needed = [k for k, v in self.GROUND_TRUTH.items() if v["archive"]]
        total = len(self.GROUND_TRUTH) * 3 + len(flag_needed) + len(reply_needed) + len(archive_needed)
        scored = (sum(1 for v in self.categories.values() if v) +
                  sum(1 for v in self.priorities.values() if v) +
                  sum(1 for v in self.labels.values() if v) +
                  sum(1 for v in self.flags.values() if v) +
                  sum(1 for v in self.replies.values() if v) +
                  sum(1 for v in self.archives.values() if v))
        return max(0.001, min(0.999, scored / total if total else 0))

    def grade(self) -> float:
        return self._grade()

    def state(self) -> dict:
        return {"task_id": self.TASK_ID, "step_count": self.step_count, "done": self.done, "grade": self._grade()}
