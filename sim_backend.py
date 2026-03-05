from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sim_env import BOX_POSE, make_sim_env


class SimBackend(ABC):
    """Abstract simulation backend for policy rollout/evaluation."""

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_initial_object_pose(self, object_pose: np.ndarray) -> None:
        raise NotImplementedError


class MuJoCoSimBackend(SimBackend):
    """MuJoCo-backed implementation of SimBackend."""

    def __init__(self, task_name: str, equipment_model: str):
        self._env = make_sim_env(task_name, equipment_model)

    def reset(self):
        return self._env.reset()

    def step(self, action: np.ndarray):
        return self._env.step(action)

    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        return self._env._physics.render(height=height, width=width, camera_id=camera_id)

    @property
    def max_reward(self) -> float:
        return self._env.task.max_reward

    def set_initial_object_pose(self, object_pose: np.ndarray) -> None:
        BOX_POSE[0] = object_pose


def make_sim_backend(task_name: str, equipment_model: str, backend: str = "mujoco") -> SimBackend:
    if backend == "mujoco":
        return MuJoCoSimBackend(task_name=task_name, equipment_model=equipment_model)
    raise NotImplementedError(f"Unsupported sim backend: {backend}")
