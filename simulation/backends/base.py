from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from PACT.core.types import Observation, RewardMeta, StepResult


class SimulationBackend(ABC):
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

