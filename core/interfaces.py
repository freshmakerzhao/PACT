from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import Observation, RewardMeta, StepResult


class Backend(ABC):
    @abstractmethod
    def reset(self) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_obs(self) -> Observation:
        raise NotImplementedError

    @property
    @abstractmethod
    def reward_meta(self) -> RewardMeta:
        raise NotImplementedError

    @abstractmethod
    def set_scene(self, scene_state: Optional[object]) -> None:
        raise NotImplementedError


class Policy(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Observation) -> np.ndarray:
        raise NotImplementedError


class TestbedRunner(ABC):
    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

