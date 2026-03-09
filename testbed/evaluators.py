"""Excavator pose flow evaluation. All thresholds live in excavator_pose_reward."""

from __future__ import annotations

from typing import Dict

import numpy as np

from PACT.simulation.rewards import ExcavatorPoseReward


def evaluate_excavator_pose_flow(joint_traj: np.ndarray) -> Dict[str, object]:
    evaluator = ExcavatorPoseReward()
    result = evaluator.evaluate(joint_traj)
    return {
        "pose_reward": result.reward,
        "pose_phases": result.phases,
        "pose_phase_flags": result.phase_flags,
        "pose_phase_switch_steps": result.phase_switch_steps,
    }

