"""
Task 2 — Medium: Prioritize and Label Inbox + Draft Reply
Given 5 emails: assign priority (high/medium/low), apply labels, draft a reply to the most urgent.
Multi-dimensional scoring: priority accuracy, label accuracy, reply quality.
"""

from typing import Dict, Tuple
from env.models import (
    Email, Observation, Action, Reward, TaskInfo,
    VALID_PRIORITIES, VALID_LABELS
)
from env.data import (
    TASK2_EMAILS, TASK2_GROUND_TRUTH,
    TASK2_REPLY_TARGET, TASK2_REPLY_KEYWORDS
)


class Task2PrioritizeAndLabel:
    """
    Medium task: 5-email inbox requiring priority assignment, label assignment,
    and a contextual reply to the most urgent email.

    Scoring breakdown (total = 1.0):
      - Priority correct per email: 5 × 0.1 = 0.50
      - Label correct per email:    5 × 0.1 = 0.50 → scaled to 0.30
      - Reply quality:              keyword coverage → max 0.20
    Normalized to [0.0, 1.0].
    """

    TASK_ID = "prioritize_and_label"
    TASK_NAME = "Prioritize and Label Inbox"
    DIFFICULTY = "medium"
    MAX_STEPS = 20
    DESCRIPTION = (
        "You have 5 emails. For each email:\n"
        "1. Set priority: action_type='prioritize', value='high'|'medium'|'low'\n"
        "2. Apply label: action_type='label', value='action_required'|'fyi'|'waiting'|'resolved'\n"
        "Then draft a reply to the most urgent email: action_type='reply', value='<reply text>'"
    )
    OBJECTIVES = [
        "Assign correct priority to all 5 emails",
        "Apply correct label to all 5 emails",
        f"Draft a relevant reply to the overdue invoice email ({TASK2_REPLY_TARGET})",
    ]
    HINTS = [
        "Overdue invoices and mandatory security actions are HIGH priority.",
        "Social invites and FYI notices are LOW priority.",
        "Reply to the invoice email — acknowledge and confirm payment status.",
    ]

    # Scoring weights
    W_PRIORITY = 0.50
    W_LABEL = 0.30
    W_REPLY = 0.20

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self) -> Observation:
        self._emails: Dict[str, Email] = {
            e["id"]: Email(**e) for e in TASK2_EMAILS
        }
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._priority_scores: Dict[str, float] = {}
        self._label_scores: Dict[str, float] = {}
        self._reply_score: float = 0.0
        self._reply_draft: str = ""
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        if self._done:
            obs = self._build_observation(error="Episode done. Call reset().")
            return obs, Reward(value=0.0, cumulative=self._total_reward, reason="Done"), True

        self._step_count += 1
        reward = self._apply_action(action)
        self._total_reward = min(1.0, self._total_reward + reward.value)
        reward.cumulative = self._total_reward

        if action.action_type == "done" or self._step_count >= self.MAX_STEPS:
            self._done = True

        obs = self._build_observation()
        obs.done = self._done
        return obs, reward, self._done

    def _apply_action(self, action: Action) -> Reward:
        if action.action_type == "done":
            return Reward(value=0.0, cumulative=0.0, reason="Episode ended by agent.")

        if action.action_type == "skip":
            return Reward(value=0.0, cumulative=0.0, reason="Skipped.")

        email_id = action.email_id
        if not email_id or email_id not in self._emails:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Unknown email_id '{email_id}'. Valid: {list(self._emails.keys())}"
            )

        if action.action_type == "prioritize":
            if action.value not in VALID_PRIORITIES:
                return Reward(value=0.0, cumulative=0.0,
                              reason=f"Invalid priority '{action.value}'. Use: {VALID_PRIORITIES}")
            if email_id in self._priority_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} priority already set.", penalty=0.02)

            correct = TASK2_GROUND_TRUTH[email_id]["priority"]
            is_correct = (action.value == correct)
            score_contribution = (self.W_PRIORITY / len(self._emails)) if is_correct else 0.0
            self._priority_scores[email_id] = score_contribution
            self._emails[email_id].priority = action.value

            return Reward(
                value=round(score_contribution, 4),
                cumulative=0.0,
                breakdown={f"priority_{email_id}": score_contribution},
                reason=(
                    f"Priority '{action.value}' for {email_id} — "
                    f"{'correct ✓' if is_correct else f'incorrect ✗ (expected {correct})'}"
                ),
            )

        elif action.action_type == "label":
            if action.value not in VALID_LABELS:
                return Reward(value=0.0, cumulative=0.0,
                              reason=f"Invalid label '{action.value}'. Use: {VALID_LABELS}")
            if email_id in self._label_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} label already set.", penalty=0.02)

            correct = TASK2_GROUND_TRUTH[email_id]["label"]
            is_correct = (action.value == correct)
            score_contribution = (self.W_LABEL / len(self._emails)) if is_correct else 0.0
            self._label_scores[email_id] = score_contribution
            self._emails[email_id].label = action.value

            return Reward(
                value=round(score_contribution, 4),
                cumulative=0.0,
                breakdown={f"label_{email_id}": score_contribution},
                reason=(
                    f"Label '{action.value}' for {email_id} — "
                    f"{'correct ✓' if is_correct else f'incorrect ✗ (expected {correct})'}"
                ),
            )

        elif action.action_type == "reply":
            if email_id != TASK2_REPLY_TARGET:
                return Reward(
                    value=0.0, cumulative=0.0,
                    reason=f"Reply sent to wrong email. The urgent email requiring reply is '{TASK2_REPLY_TARGET}'.",
                    penalty=0.05
                )
            if self._reply_score > 0:
                return Reward(value=0.0, cumulative=0.0, reason="Reply already drafted.", penalty=0.02)

            reply_text = (action.value or "").lower()
            hits = sum(1 for kw in TASK2_REPLY_KEYWORDS if kw in reply_text)
            coverage = hits / len(TASK2_REPLY_KEYWORDS)
            score_contribution = round(coverage * self.W_REPLY, 4)
            self._reply_score = score_contribution
            self._reply_draft = action.value or ""
            self._emails[email_id].reply_draft = action.value

            return Reward(
                value=score_contribution,
                cumulative=0.0,
                breakdown={"reply_quality": score_contribution},
                reason=(
                    f"Reply drafted. Keyword coverage: {hits}/{len(TASK2_REPLY_KEYWORDS)} "
                    f"({coverage:.0%}) → score {score_contribution:.2f}"
                ),
            )

        else:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Action '{action.action_type}' not applicable in Task 2. Use prioritize/label/reply."
            )

    def _build_observation(self, error: str = None) -> Observation:
        actions_taken = []
        for eid in self._emails:
            parts = []
            if eid in self._priority_scores:
                parts.append(f"priority={self._emails[eid].priority}")
            if eid in self._label_scores:
                parts.append(f"label={self._emails[eid].label}")
            if self._emails[eid].reply_draft:
                parts.append("reply=drafted")
            if parts:
                actions_taken.append(f"{eid}: {', '.join(parts)}")

        return Observation(
            inbox=list(self._emails.values()),
            current_task=self.TASK_ID,
            step_count=self._step_count,
            max_steps=self.MAX_STEPS,
            task_info=TaskInfo(
                task_id=self.TASK_ID,
                task_name=self.TASK_NAME,
                difficulty=self.DIFFICULTY,
                description=self.DESCRIPTION,
                objectives=self.OBJECTIVES,
                actions_taken=actions_taken,
                score_so_far=self._total_reward,
                hints=self.HINTS,
            ),
            last_action_error=error,
            done=self._done,
        )

    def grade(self) -> float:
        """Final grade normalized to [0.0, 1.0]."""
        p = sum(self._priority_scores.values())
        l = sum(self._label_scores.values())
        r = self._reply_score
        return round(min(1.0, p + l + r), 4)

    def state(self) -> dict:
        return {
            "task_id": self.TASK_ID,
            "step_count": self._step_count,
            "done": self._done,
            "reward_so_far": self._total_reward,
            "priority_scores": self._priority_scores,
            "label_scores": self._label_scores,
            "reply_score": self._reply_score,
            "grade": self.grade(),
        }
