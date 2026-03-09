"""Pose-only excavator reward. Thresholds centralized here, evaluators use this module only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Target flow phases (阶段命名)
PHASES_ORDER = (
    "start",
    "bucket_out",
    "lower_arm",
    "bucket_in_scoop",
    "lift_balance",
    "swing_to_target",
    "dump_open",
)

# Centralized thresholds (容忍区间，可调参)
THRESH_SWING_DUMP = 1.0  # abs(swing) >= 进入卸料侧
THRESH_BUCKET_SCOOP = -0.6  # bucket < 铲斗切入
THRESH_BUCKET_LOADED = -1.0  # bucket < 装载成立
THRESH_BUCKET_SWING_CARRY = -0.4  # bucket < 回转运输
# Calibrated for current excavator_simple IK / task-space scripted policy:
# - during dump, bucket joint does not reach positive values; it typically opens to around ~ -0.21.
THRESH_BUCKET_DUMP = -0.55  # bucket >= 卸料动作（开始开斗）
THRESH_BUCKET_DUMPED = -0.31  # bucket >= 卸料完成


@dataclass
class PoseRewardResult:
    reward: float
    phases: List[str]
    phase_flags: Dict[str, bool]
    phase_switch_steps: Dict[str, int]


class ExcavatorPoseReward:
    """Pose-only reward for excavator dig-dump flow (no contact dependency)."""

    def __init__(self):
        self._reset_flags()

    def _reset_flags(self) -> None:
        self._flags = {
            "bucket_out": False,
            "bucket_in_scoop": False,
            "lift_balance": False,
            "swing_to_target": False,
            "dump_open": False,
        }
        self._phase_switch_steps: Dict[str, int] = {}
        self._max_reward = 0.0

    def evaluate(self, qpos_traj: np.ndarray) -> PoseRewardResult:
        self._reset_flags()
        traj = np.asarray(qpos_traj)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        phases: List[str] = []
        reward = 0.0
        for step, qpos in enumerate(traj):
            phase, phase_reward = self._phase_from_qpos(qpos)
            phases.append(phase)
            reward = max(reward, phase_reward)
            if phase not in self._phase_switch_steps:
                self._phase_switch_steps[phase] = step
        return PoseRewardResult(
            reward=reward,
            phases=phases,
            phase_flags=self._flags.copy(),
            phase_switch_steps=dict(self._phase_switch_steps),
        )

    def _phase_from_qpos(self, qpos: np.ndarray) -> Tuple[str, float]:
        swing = float(qpos[0])
        bucket = float(qpos[3])
        abs_swing = abs(swing)

        if abs_swing < THRESH_SWING_DUMP and bucket < THRESH_BUCKET_SCOOP:
            self._flags["bucket_in_scoop"] = True
            if bucket < THRESH_BUCKET_LOADED:
                self._flags["lift_balance"] = True
                return "bucket_in_scoop", 2.0
            self._flags["bucket_out"] = True
            return "bucket_out", 1.0

        if abs_swing >= THRESH_SWING_DUMP and bucket < THRESH_BUCKET_SWING_CARRY:
            self._flags["swing_to_target"] = True
            return "swing_to_target", 2.8

        if abs_swing >= THRESH_SWING_DUMP and bucket >= THRESH_BUCKET_DUMPED:
            self._flags["dump_open"] = True
            return "dump_open", 4.0
        if abs_swing >= THRESH_SWING_DUMP and bucket >= THRESH_BUCKET_DUMP:
            self._flags["swing_to_target"] = True
            return "dump_open", 3.0

        return "start", 0.0

    def reset(self) -> None:
        """Reset for new episode (call before first step_reward of episode)."""
        self._reset_flags()

    def step_reward(self, qpos: np.ndarray) -> float:
        """Single-step reward for env integration. Returns cumulative max for episode so far."""
        _, r = self._phase_from_qpos(qpos)
        self._max_reward = max(self._max_reward, r)
        return self._max_reward

