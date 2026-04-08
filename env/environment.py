from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from env.models import Action, Observation, Reward
from env.tasks.task1_easy import Task1Easy
from env.tasks.task2_medium import Task2Medium
from env.tasks.task3_hard import Task3Hard
from env.data import get_task_emails

TASK_REGISTRY = {
    Task1Easy.TASK_ID: Task1Easy,
    Task2Medium.TASK_ID: Task2Medium,
    Task3Hard.TASK_ID: Task3Hard,
}
ALL_TASK_IDS = list(TASK_REGISTRY.keys())

class EmailTriageEnv:
    def __init__(self):
        self._task = None
        self._task_id = None
        self._seed = 42
        self._episode_rewards = []

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Available: {ALL_TASK_IDS}")
        self._task_id = task_id
        self._seed = seed
        emails = get_task_emails(task_id)
        self._task = TASK_REGISTRY[task_id](emails=emails, seed=seed)
        self._episode_rewards = []
        return self._task.reset()

    def step(self, action: Action):
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        obs, reward, done = self._task.step(action)
        self._episode_rewards.append(reward.value)
        info = {"task_id": self._task_id, "step": obs.step_count,
                "cumulative_reward": getattr(reward, 'cumulative', reward.value),
                "grade": self._task.grade() if done else None}
        return obs, reward, done, info

    def state(self) -> dict:
        if self._task is None:
            return {"status": "not_initialized", "available_tasks": ALL_TASK_IDS}
        return {**self._task.state(), "episode_rewards": self._episode_rewards, "available_tasks": ALL_TASK_IDS}

    def grade(self) -> float:
        return self._task.grade() if self._task else 0.0

    def close(self):
        self._task = None
        self._task_id = None
        self._episode_rewards = []
