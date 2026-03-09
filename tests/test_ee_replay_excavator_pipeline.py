"""Fixed-scene regression for excavator ee_replay_pipeline: replay must reach pose_reward 4.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure PACT and project root on path
for _p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PACT.testbed.pipelines import ee_replay_pipeline
from PACT.policies.registry import PolicyRegistry


def test_ee_replay_excavator_fixed_pose_replay_reward_4():
    """With fixed box pose, excavator ee_replay replay must reach pose_reward 4.0 and dump_open."""
    fixed_pose = np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    policy_cls = PolicyRegistry().get("sim_lifting_cube_scripted", "excavator_simple")
    _, info = ee_replay_pipeline(
        task_name="sim_lifting_cube_scripted",
        equipment_model="excavator_simple",
        episode_len=400,
        policy_cls=policy_cls,
        render_cam=None,
        fixed_excavator_box_pose=fixed_pose,
    )
    pose_eval = info.get("pose_eval") or {}
    ee_pose_eval = info.get("ee_pose_eval") or {}
    replay_joint_traj = info.get("replay_joint_traj")
    assert replay_joint_traj is not None
    assert pose_eval.get("pose_reward") == 4.0, (
        f"pose_eval should come from replay and reach 4.0, got {pose_eval.get('pose_reward')}"
    )
    assert pose_eval.get("pose_phase_flags", {}).get("dump_open") is True
    assert pose_eval.get("pose_phase_flags", {}).get("swing_to_target") is True
    assert "replay_joint_traj" in info
    assert "ee_pose_eval" in info
    assert ee_pose_eval.get("pose_reward") is not None
