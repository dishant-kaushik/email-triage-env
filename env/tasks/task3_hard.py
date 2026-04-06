"""
Task 3 — Hard: Full Inbox Management
Manage a 10-email inbox: classify, prioritize, label, detect duplicate threads,
identify action items, draft contextual replies, archive resolved/noise emails.

Scoring breakdown (total = 1.0):
  - Category correct:     10 × 0.03 = 0.30
  - Priority correct:     10 × 0.02 = 0.20
  - Label correct:        10 × 0.01 = 0.10
  - Duplicate detection:  1  × 0.10 = 0.10
  - Action items flagged: 5  × 0.02 = 0.10
  - Reply quality:        3 emails × 0.05 = 0.15 (keyword coverage)
  - Archiving noise:      archive spam+newsletter+resolved = 0.05
Total = 1.00 (each component capped at its weight)
"""

from typing import Dict, Set, Tuple
from env.models import (
    Email, Observation, Action, Reward, TaskInfo,
    VALID_CATEGORIES, VALID_PRIORITIES, VALID_LABELS
)
from env.data import (
    TASK3_EMAILS, TASK3_GROUND_TRUTH,
    TASK3_DUPLICATE_THREAD, TASK3_ACTION_ITEMS,
    TASK3_REPLY_KEYWORDS
)


class Task3FullInboxManagement:
    """
    Hard task: comprehensive 10-email inbox management with multi-dimensional grading.
    Tests classification, prioritization, labeling, deduplication, action item detection,
    reply drafting, and archiving — all in a single episode.
    """

    TASK_ID = "full_inbox_management"
    TASK_NAME = "Full Inbox Management"
    DIFFICULTY = "hard"
    MAX_STEPS = 40
    DESCRIPTION = (
        "You have 10 emails. Your job:\n"
        "1. Classify each email (category)\n"
        "2. Assign priority (high/medium/low)\n"
        "3. Apply label (action_required/fyi/waiting/resolved/duplicate/archived)\n"
        "4. Flag emails that need action: action_type='flag', email_id=<id>\n"
        "5. Draft replies for urgent emails needing response (action_type='reply')\n"
        "6. Archive spam, newsletters, and resolved emails (action_type='archive')\n"
        "Note: Emails t3-e003 and t3-e004 are follow-ups to t3-e001 — same thread."
    )
    OBJECTIVES = [
        "Correctly classify all 10 emails",
        "Assign correct priority to all 10 emails",
        "Apply correct label to all 10 emails",
        "Detect the duplicate thread (t3-e003, t3-e004 duplicate t3-e001)",
        "Flag all 5 action-required emails",
        "Draft contextual replies to urgent emails needing responses",
        "Archive spam, newsletters, and auto-resolved emails",
    ]
    HINTS = [
        "Emails from CEO and CI/CD pipeline failures are always high priority.",
        "Thread IDs help identify duplicate chains — look for thread_id fields.",
        "Flag t3-e001, t3-e003, t3-e005, t3-e008, t3-e010 as action items.",
        "Spam and newsletters should be archived immediately.",
        "A good reply to a board presentation request mentions: slides, deadline, board.",
    ]

    # Score weights
    W_CATEGORY = 0.30
    W_PRIORITY = 0.20
    W_LABEL = 0.10
    W_DUPLICATE = 0.10
    W_FLAG = 0.10
    W_REPLY = 0.15
    W_ARCHIVE = 0.05

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self) -> Observation:
        self._emails: Dict[str, Email] = {
            e["id"]: Email(**e) for e in TASK3_EMAILS
        }
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

        self._category_scores: Dict[str, float] = {}
        self._priority_scores: Dict[str, float] = {}
        self._label_scores: Dict[str, float] = {}
        self._flagged: Set[str] = set()
        self._reply_scores: Dict[str, float] = {}
        self._archived: Set[str] = set()
        self._duplicate_detected: bool = False

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
        if action.action_type in ("done", "skip"):
            return Reward(value=0.0, cumulative=0.0, reason="Action acknowledged.")

        email_id = action.email_id
        if email_id and email_id not in self._emails:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Unknown email_id '{email_id}'. Valid: {list(self._emails.keys())}"
            )

        atype = action.action_type

        # --- CLASSIFY ---
        if atype == "classify":
            if action.value not in VALID_CATEGORIES:
                return Reward(value=0.0, cumulative=0.0,
                              reason=f"Invalid category. Use: {VALID_CATEGORIES}")
            if email_id in self._category_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} already classified.", penalty=0.01)
            correct = TASK3_GROUND_TRUTH[email_id]["category"]
            is_correct = (action.value == correct)
            unit = self.W_CATEGORY / len(self._emails)
            contrib = unit if is_correct else 0.0
            self._category_scores[email_id] = contrib
            self._emails[email_id].category = action.value
            return Reward(
                value=round(contrib, 4), cumulative=0.0,
                breakdown={f"category_{email_id}": contrib},
                reason=f"Category '{action.value}' for {email_id} — {'✓ correct' if is_correct else f'✗ expected {correct}'}"
            )

        # --- PRIORITIZE ---
        elif atype == "prioritize":
            if action.value not in VALID_PRIORITIES:
                return Reward(value=0.0, cumulative=0.0,
                              reason=f"Invalid priority. Use: {VALID_PRIORITIES}")
            if email_id in self._priority_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} priority already set.", penalty=0.01)
            correct = TASK3_GROUND_TRUTH[email_id]["priority"]
            is_correct = (action.value == correct)
            unit = self.W_PRIORITY / len(self._emails)
            contrib = unit if is_correct else 0.0
            self._priority_scores[email_id] = contrib
            self._emails[email_id].priority = action.value
            return Reward(
                value=round(contrib, 4), cumulative=0.0,
                breakdown={f"priority_{email_id}": contrib},
                reason=f"Priority '{action.value}' for {email_id} — {'✓' if is_correct else f'✗ expected {correct}'}"
            )

        # --- LABEL ---
        elif atype == "label":
            if action.value not in VALID_LABELS:
                return Reward(value=0.0, cumulative=0.0,
                              reason=f"Invalid label. Use: {VALID_LABELS}")
            if email_id in self._label_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} label already set.", penalty=0.01)
            correct = TASK3_GROUND_TRUTH[email_id]["label"]
            is_correct = (action.value == correct)
            unit = self.W_LABEL / len(self._emails)
            contrib = unit if is_correct else 0.0
            self._label_scores[email_id] = contrib
            self._emails[email_id].label = action.value

            # Bonus: detect the duplicate thread
            # Agent earns the duplicate bonus if they label the threaded email (t3-e003 or t3-e004)
            # as 'duplicate' OR correctly note thread membership via any label on t3-e003/t3-e004
            email_thread = self._emails[email_id].thread_id
            if not self._duplicate_detected and email_thread == TASK3_DUPLICATE_THREAD:
                if action.value in ("duplicate", "fyi", "action_required"):
                    self._duplicate_detected = True
                    bonus = self.W_DUPLICATE
                    contrib += bonus
                    return Reward(
                        value=round(contrib, 4), cumulative=0.0,
                        breakdown={f"label_{email_id}": unit, "duplicate_detection": bonus},
                        reason=f"Label '{action.value}' for {email_id} ({'✓' if is_correct else '~'}) + duplicate thread detected! +{bonus} bonus."
                    )

            return Reward(
                value=round(contrib, 4), cumulative=0.0,
                breakdown={f"label_{email_id}": contrib},
                reason=f"Label '{action.value}' for {email_id} — {'✓' if is_correct else f'✗ expected {correct}'}"
            )

        # --- FLAG ---
        elif atype == "flag":
            if email_id in self._flagged:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} already flagged.", penalty=0.01)
            self._flagged.add(email_id)
            self._emails[email_id].flagged = True
            is_correct = email_id in TASK3_ACTION_ITEMS
            unit = self.W_FLAG / len(TASK3_ACTION_ITEMS)
            contrib = unit if is_correct else 0.0
            penalty = 0.02 if not is_correct else 0.0
            return Reward(
                value=round(contrib, 4), cumulative=0.0,
                breakdown={f"flag_{email_id}": contrib},
                reason=(
                    f"Flagged {email_id} — {'✓ is an action item' if is_correct else '✗ not an action item (false positive)'}."
                ),
                penalty=penalty
            )

        # --- REPLY ---
        elif atype == "reply":
            if email_id not in TASK3_REPLY_KEYWORDS:
                return Reward(
                    value=0.0, cumulative=0.0,
                    reason=f"{email_id} does not require a reply. Reply-required: {list(TASK3_REPLY_KEYWORDS.keys())}",
                    penalty=0.02
                )
            if email_id in self._reply_scores:
                return Reward(value=0.0, cumulative=0.0, reason=f"Reply for {email_id} already drafted.", penalty=0.01)
            reply_text = (action.value or "").lower()
            keywords = TASK3_REPLY_KEYWORDS[email_id]
            hits = sum(1 for kw in keywords if kw in reply_text)
            coverage = hits / len(keywords)
            unit = self.W_REPLY / len(TASK3_REPLY_KEYWORDS)
            contrib = round(coverage * unit, 4)
            self._reply_scores[email_id] = contrib
            self._emails[email_id].reply_draft = action.value
            return Reward(
                value=contrib, cumulative=0.0,
                breakdown={f"reply_{email_id}": contrib},
                reason=f"Reply for {email_id}: {hits}/{len(keywords)} keywords hit → score {contrib:.3f}"
            )

        # --- ARCHIVE ---
        elif atype == "archive":
            if email_id in self._archived:
                return Reward(value=0.0, cumulative=0.0, reason=f"{email_id} already archived.", penalty=0.01)
            self._archived.add(email_id)
            self._emails[email_id].archived = True
            gt = TASK3_GROUND_TRUTH[email_id]
            should_archive = gt.get("archive", False)
            archive_emails = [eid for eid, v in TASK3_GROUND_TRUTH.items() if v.get("archive")]
            unit = self.W_ARCHIVE / len(archive_emails)
            contrib = unit if should_archive else 0.0
            penalty = 0.03 if not should_archive else 0.0
            return Reward(
                value=round(contrib, 4), cumulative=0.0,
                breakdown={f"archive_{email_id}": contrib},
                reason=(
                    f"Archived {email_id} — {'✓ correct' if should_archive else '✗ should not be archived'}."
                ),
                penalty=penalty
            )

        else:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Unknown action '{atype}'. Use: classify/prioritize/label/flag/reply/archive/done."
            )

    def _build_observation(self, error: str = None) -> Observation:
        actions_taken = []
        for eid, email in self._emails.items():
            parts = []
            if eid in self._category_scores: parts.append(f"cat={email.category}")
            if eid in self._priority_scores: parts.append(f"pri={email.priority}")
            if eid in self._label_scores:    parts.append(f"lbl={email.label}")
            if eid in self._flagged:         parts.append("flagged")
            if eid in self._reply_scores:    parts.append("replied")
            if eid in self._archived:        parts.append("archived")
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
        cat = sum(self._category_scores.values())
        pri = sum(self._priority_scores.values())
        lbl = sum(self._label_scores.values())
        dup = self.W_DUPLICATE if self._duplicate_detected else 0.0
        flg = sum(self.W_FLAG / len(TASK3_ACTION_ITEMS) for eid in self._flagged if eid in TASK3_ACTION_ITEMS)
        rep = sum(self._reply_scores.values())
        archive_correct = [eid for eid in self._archived if TASK3_GROUND_TRUTH[eid].get("archive")]
        arc = len(archive_correct) * (self.W_ARCHIVE / len([e for e, v in TASK3_GROUND_TRUTH.items() if v.get("archive")]))
        total = cat + pri + lbl + dup + flg + rep + arc
        return round(min(1.0, total), 4)

    def state(self) -> dict:
        return {
            "task_id": self.TASK_ID,
            "step_count": self._step_count,
            "done": self._done,
            "reward_so_far": self._total_reward,
            "grade": self.grade(),
            "breakdown": {
                "category": sum(self._category_scores.values()),
                "priority": sum(self._priority_scores.values()),
                "label": sum(self._label_scores.values()),
                "duplicate_detected": self._duplicate_detected,
                "flags": len([e for e in self._flagged if e in TASK3_ACTION_ITEMS]),
                "replies": len(self._reply_scores),
                "archived": len(self._archived),
            },
        }
