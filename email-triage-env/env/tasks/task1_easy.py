"""
Task 1 — Easy: Email Classification
Classify 3 emails into correct categories.
Partial credit per correct classification.
"""

from typing import Dict, Tuple
from env.models import (
    Email, Observation, Action, Reward, TaskInfo,
    VALID_CATEGORIES
)
from env.data import TASK1_EMAILS, TASK1_GROUND_TRUTH


class Task1ClassifyEmails:
    """
    Easy task: classify each of 3 emails (spam / urgent / newsletter / normal).
    Agent earns 1/3 reward per correct classification.
    Episode ends when all emails classified or max_steps reached.
    """

    TASK_ID = "classify_emails"
    TASK_NAME = "Email Classification"
    DIFFICULTY = "easy"
    MAX_STEPS = 10
    DESCRIPTION = (
        "You have 3 emails in your inbox. Classify each one into the correct "
        "category: spam, urgent, newsletter, or normal. "
        "Use action_type='classify' with the email_id and value=<category>."
    )
    OBJECTIVES = [
        "Classify t1-e001 correctly (hint: look at sender domain and prize claim)",
        "Classify t1-e002 correctly (hint: look at sender, subject urgency, impact)",
        "Classify t1-e003 correctly (hint: check for unsubscribe links, bulk sender)",
    ]
    HINTS = [
        "Suspicious sender domains with prize claims are usually spam.",
        "Monitoring alerts with 'CRITICAL' and 'prod' are urgent.",
        "Emails with 'Unsubscribe' and 'newsletter' in subject are newsletters.",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self) -> Observation:
        """Initialize task state."""
        self._emails: Dict[str, Email] = {
            e["id"]: Email(**e) for e in TASK1_EMAILS
        }
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._classified_correct: Dict[str, bool] = {}
        self._classified_attempted: set = set()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        """Apply action and return (observation, reward, done)."""
        if self._done:
            obs = self._build_observation(error="Episode already done. Call reset().")
            return obs, Reward(value=0.0, cumulative=self._total_reward, reason="Episode done"), True

        self._step_count += 1
        reward, done = self._apply_action(action)
        self._total_reward = min(1.0, self._total_reward + reward.value)
        reward.cumulative = self._total_reward

        # Episode ends if all classified or done action or max steps
        if (len(self._classified_attempted) >= len(self._emails)
                or action.action_type == "done"
                or self._step_count >= self.MAX_STEPS):
            self._done = True
            done = True

        obs = self._build_observation()
        obs.done = self._done
        return obs, reward, self._done

    def _apply_action(self, action: Action) -> Tuple[Reward, bool]:
        """Evaluate the action and return a reward."""
        # Handle non-targeting actions
        if action.action_type == "done":
            remaining = len(self._emails) - len(self._classified_attempted)
            penalty = 0.05 * remaining  # small penalty for skipping emails
            return Reward(
                value=0.0,
                cumulative=0.0,
                reason=f"Episode ended by agent. {remaining} emails not classified.",
                penalty=penalty
            ), True

        if action.action_type == "skip":
            return Reward(value=0.0, cumulative=0.0, reason="Action skipped."), False

        # Validate email_id
        if not action.email_id or action.email_id not in self._emails:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Unknown email_id: {action.email_id}. Valid IDs: {list(self._emails.keys())}"
            ), False

        # Only classify is valid for task 1
        if action.action_type != "classify":
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Action '{action.action_type}' not used in Task 1. Use 'classify'."
            ), False

        if action.value not in VALID_CATEGORIES:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Invalid category '{action.value}'. Valid: {VALID_CATEGORIES}"
            ), False

        # Penalize re-classifying same email
        if action.email_id in self._classified_attempted:
            return Reward(
                value=0.0, cumulative=0.0,
                reason=f"Email {action.email_id} already classified. Moving on.",
                penalty=0.05
            ), False

        # Score this classification
        self._classified_attempted.add(action.email_id)
        correct = TASK1_GROUND_TRUTH[action.email_id]
        is_correct = (action.value == correct)
        self._classified_correct[action.email_id] = is_correct

        # Update email object
        self._emails[action.email_id].category = action.value

        if is_correct:
            step_reward = round(1.0 / len(self._emails), 4)
            return Reward(
                value=step_reward,
                cumulative=0.0,
                breakdown={action.email_id: step_reward},
                reason=f"Correct! {action.email_id} is indeed '{correct}'.",
            ), False
        else:
            return Reward(
                value=0.0,
                cumulative=0.0,
                breakdown={action.email_id: 0.0},
                reason=f"Incorrect. You said '{action.value}' but correct answer is '{correct}'.",
            ), False

    def _build_observation(self, error: str = None) -> Observation:
        actions_taken = [
            f"{'✓' if v else '✗'} {k}: classified as {self._emails[k].category}"
            for k, v in self._classified_correct.items()
        ]
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
        """Final grader: returns score in [0.0, 1.0]."""
        if not self._classified_attempted:
            return 0.0
        correct_count = sum(1 for v in self._classified_correct.values() if v)
        return round(correct_count / len(TASK1_GROUND_TRUTH), 4)

    def state(self) -> dict:
        return {
            "task_id": self.TASK_ID,
            "step_count": self._step_count,
            "done": self._done,
            "reward_so_far": self._total_reward,
            "classified_correct": self._classified_correct,
            "grade": self.grade(),
        }
