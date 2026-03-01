"""Software safety helpers (skeleton).

Policies:
- Dynamic limit readback (from SDK)
- Clamp actions to limits
- On anomaly: hold current position
- Logging only (print)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class SafetyState:
    last_qpos: np.ndarray | None = None


def clamp_to_limits(qpos: np.ndarray, limits: JointLimits) -> np.ndarray:
    """Clamp joint positions to dynamic limits."""
    # 支持 qpos 与 limits 维度不一致（例如真实机械臂6轴，但训练state_dim为7含夹爪）
    min_len = min(qpos.shape[0], limits.lower.shape[0], limits.upper.shape[0])
    clipped = np.minimum(
        np.maximum(qpos[:min_len], limits.lower[:min_len]),
        limits.upper[:min_len],
    )
    if qpos.shape[0] == min_len:
        return clipped
    # 若 qpos 更长，保留后续维度不变
    out = qpos.copy()
    out[:min_len] = clipped
    return out


def detect_large_jump(target_qpos: np.ndarray, current_qpos: np.ndarray, max_delta: float) -> bool:
    """Simple jump detector based on per-joint delta threshold (rad)."""
    min_len = min(target_qpos.shape[0], current_qpos.shape[0])
    return np.any(np.abs(target_qpos[:min_len] - current_qpos[:min_len]) > max_delta)


def hold_position(current_qpos: np.ndarray) -> np.ndarray:
    """Hold current position on anomaly."""
    return current_qpos.copy()
