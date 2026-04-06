"""
EmailTriageEnv — Main environment class.
Implements the OpenEnv interface: reset(), step(), state().
Dispatches to the correct task based on task_id.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from env.models import (
    Action, Observation, Reward, StepResponse, StateResponse
)
from env.tasks.task1_easy import Task1ClassifyEmails
from env.tasks.task2_medium import Task2PrioritizeAndLabel
from env.tasks.task3_hard import Task3FullInboxManagement

TASK_REGISTRY = {
    Task1ClassifyEmails.TASK_ID: Task1ClassifyEmails,
    Task2PrioritizeAndLabel.TASK_ID: Task2PrioritizeAndLabel,
    Task3FullInboxManagement.TASK_ID: Task3FullInboxManagement,
}

ALL_TASK_IDS = list(TASK_REGISTRY.keys())


class EmailTriageEnv:
    """
    OpenEnv-compliant environment for email inbox management.

    Usage:
        env = EmailTriageEnv()
        obs = env.reset("classify_emails")
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        print(env.grade())
    """

    def __init__(self):
        self._task = None
        self._task_id: Optional[str] = None
        self._seed: int = 42
        self._episode_rewards: list = []

    # ------------------------------------------------------------------
    # Core OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """
        Initialize a new episode for the specified task.
        Returns the initial observation.
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {ALL_TASK_IDS}"
            )
        self._task_id = task_id
        self._seed = seed
        self._task = TASK_REGISTRY[task_id](seed=seed)
        self._episode_rewards = []
        return self._task.reset()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action to the environment.
        Returns (observation, reward, done, info).
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        obs, reward, done = self._task.step(action)
        self._episode_rewards.append(reward.value)

        info = {
            "task_id": self._task_id,
            "step": obs.step_count,
            "cumulative_reward": reward.cumulative,
            "grade": self._task.grade() if done else None,
            "reward_breakdown": reward.breakdown,
        }
        return obs, reward, done, info

    def state(self) -> dict:
        """Returns the current environment state."""
        if self._task is None:
            return {"status": "not_initialized", "available_tasks": ALL_TASK_IDS}
        return {
            **self._task.state(),
            "episode_rewards": self._episode_rewards,
            "available_tasks": ALL_TASK_IDS,
        }

    def grade(self) -> float:
        """Returns the final episode grade [0.0, 1.0]."""
        if self._task is None:
            return 0.0
        return self._task.grade()

    def close(self):
        """Clean up resources."""
        self._task = None
        self._task_id = None
        self._episode_rewards = []
