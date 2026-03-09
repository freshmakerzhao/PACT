from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class Observation:
    qpos: np.ndarray
    qvel: np.ndarray
    env_state: np.ndarray
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, key: str):
        if key == "qpos":
            return self.qpos
        if key == "qvel":
            return self.qvel
        if key == "env_state":
            return self.env_state
        if key == "images":
            return self.images
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in ("qpos", "qvel", "env_state", "images") or key in self.extras

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool = False
    info: Dict[str, object] = field(default_factory=dict)


@dataclass
class RewardMeta:
    max_reward: float
    success_threshold: Optional[float] = None

