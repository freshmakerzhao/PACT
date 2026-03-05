from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ee_sim_env import make_ee_sim_env


class EESimBackend(ABC):
    """Abstract backend for end-effector-space simulation rollout."""

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_reward(self) -> float:
        raise NotImplementedError


class MuJoCoEESimBackend(EESimBackend):
    """MuJoCo-backed implementation for EE simulation backend."""

    def __init__(self, task_name: str, equipment_model: str):
        self._env = make_ee_sim_env(task_name, equipment_model)

    def reset(self):
        return self._env.reset()

    def step(self, action: np.ndarray):
        return self._env.step(action)

    @property
    def max_reward(self) -> float:
        return self._env.task.max_reward


def make_ee_sim_backend(task_name: str, equipment_model: str, backend: str = "mujoco") -> EESimBackend:
    if backend == "mujoco":
        return MuJoCoEESimBackend(task_name=task_name, equipment_model=equipment_model)
    raise NotImplementedError(f"Unsupported ee sim backend: {backend}")

