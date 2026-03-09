from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from PACT.simulation.backends import make_ee_backend, make_sim_backend
from PACT.simulation.backends.base import SimulationBackend
from PACT.simulation.scene import SceneState
from PACT.testbed.evaluators import evaluate_excavator_pose_flow
from PACT.testbed.rollout import rollout_episode

# Excavator replay follower: tunable constants (keep at module top for calibration)
_REPLAY_QPOS_TOL = 0.12
_REPLAY_MAX_FOLLOW_STEPS_PER_TARGET = 45
_REPLAY_DUMP_HOLD_STEPS = 120
# Hold when target is in or near dump-open region so swing can catch up
_REPLAY_DUMP_WINDOW_SWING_THRESH = 0.95
_REPLAY_DUMP_WINDOW_BUCKET_THRESH = -0.35


def _replay_excavator_joint_traj(
    sim_backend: SimulationBackend,
    ee_joint_traj: np.ndarray,
    qpos_tol: float = _REPLAY_QPOS_TOL,
    max_follow_steps_per_target: int = _REPLAY_MAX_FOLLOW_STEPS_PER_TARGET,
    dump_hold_steps: int = _REPLAY_DUMP_HOLD_STEPS,
    dump_swing_thresh: float = _REPLAY_DUMP_WINDOW_SWING_THRESH,
    dump_bucket_thresh: float = _REPLAY_DUMP_WINDOW_BUCKET_THRESH,
) -> Tuple[List[object], np.ndarray, np.ndarray, Dict[str, object]]:
    """Replay EE excavator trajectory on joint-space sim with per-target following.

    Each target from ee_joint_traj is held until qpos error is below qpos_tol or
    max_follow_steps_per_target is reached. In the dump window (abs(swing) >= dump_swing_thresh
    and bucket >= dump_bucket_thresh), we hold for an extra dump_hold_steps so swing and
    bucket can align for reward 4.0.

    Returns:
        replay_episode: list of timesteps (first is reset, then one per step)
        replay_joint_traj: (T, 4) actual qpos at each step (same length as steps)
        action_traj: (T+1, 4) ctrl sent each step; last row duplicates second-to-last for HDF5
        follow_stats: dict with steps_per_target, mean_errors, dump_hold_count (for diagnostics)
    """
    replay_episode: List[object] = []
    action_list: List[np.ndarray] = []
    steps_per_target: List[int] = []
    errors_per_target: List[float] = []
    dump_hold_count = 0

    ts = sim_backend.reset()
    replay_episode.append(ts)
    n_targets = len(ee_joint_traj)
    if n_targets == 0:
        return replay_episode, np.zeros((0, 4)), np.zeros((0, 4)), {}

    # Find best dump-open target (abs(swing) >= thresh, bucket as high as possible) for final hold
    ee_arr = np.asarray(ee_joint_traj).reshape(n_targets, -1)
    if ee_arr.shape[1] < 4:
        best_dump_target = None
    else:
        abs_sw = np.abs(ee_arr[:, 0])
        bkt = ee_arr[:, 3]
        in_region = (abs_sw >= dump_swing_thresh) & (bkt >= dump_bucket_thresh)
        if np.any(in_region):
            idx = np.where(in_region)[0]
            best_idx = idx[np.argmax(bkt[idx])]
            best_dump_target = np.asarray(ee_arr[best_idx], dtype=np.float64).flat[:4].copy()
        else:
            best_dump_target = None

    for i in range(n_targets):
        target = np.asarray(ee_joint_traj[i], dtype=np.float64)
        if target.size != 4:
            target = target.flat[:4]
        in_dump_window = (
            abs(float(target[0])) >= dump_swing_thresh
            and float(target[3]) >= dump_bucket_thresh
        )
        steps_used = 0
        errs: List[float] = []

        while steps_used < max_follow_steps_per_target:
            ts = sim_backend.step(target)
            replay_episode.append(ts)
            action_list.append(target.copy())
            steps_used += 1
            curr = np.asarray(ts.observation["qpos"], dtype=np.float64).flat[:4]
            err = float(np.linalg.norm(curr - target))
            errs.append(err)
            if err <= qpos_tol:
                break

        if in_dump_window and steps_used > 0:
            for _ in range(dump_hold_steps):
                ts = sim_backend.step(target)
                replay_episode.append(ts)
                action_list.append(target.copy())
                steps_used += 1
            dump_hold_count += 1

        steps_per_target.append(steps_used)
        errors_per_target.append(float(np.mean(errs)) if errs else 0.0)

    # Final dump hold: hold best dump target so swing and bucket align for reward 4.0
    hold_target = best_dump_target
    if hold_target is None and ee_arr.shape[0] > 0 and ee_arr.shape[1] >= 4:
        second_half = ee_arr[len(ee_arr) // 2 :]
        best_idx = np.argmax(second_half[:, 3])
        hold_target = np.asarray(second_half[best_idx], dtype=np.float64).flat[:4].copy()
    if hold_target is not None:
        for _ in range(dump_hold_steps):
            ts = sim_backend.step(hold_target)
            replay_episode.append(ts)
            action_list.append(hold_target.copy())
        dump_hold_count += 1

    replay_joint_traj = np.array([ts.observation["qpos"] for ts in replay_episode[1:]])
    action_arr = np.array(action_list, dtype=np.float64)
    if action_arr.size > 0:
        last_action = action_arr[-1:].copy()
        action_traj = np.concatenate([action_arr, last_action], axis=0)
    else:
        action_traj = action_arr

    follow_stats = {
        "steps_per_target": steps_per_target,
        "mean_errors": errors_per_target,
        "dump_hold_count": dump_hold_count,
        "total_replay_steps": len(action_list),
    }
    return replay_episode, replay_joint_traj, action_traj, follow_stats


def ee_replay_pipeline(
    task_name: str,
    equipment_model: str,
    episode_len: int,
    policy_cls,
    render_cam: str | None = None,
    fixed_excavator_box_pose: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    ee_backend = make_ee_backend(task_name, equipment_model=equipment_model, backend="mujoco")
    sim_backend = make_sim_backend(task_name, equipment_model=equipment_model, backend="mujoco")

    if fixed_excavator_box_pose is not None:
        scene_state = SceneState(box_pose=fixed_excavator_box_pose)
        ee_backend.set_scene(scene_state)
        sim_backend.set_scene(scene_state)

    policy = policy_cls(False)
    episode, _ = rollout_episode(
        backend=ee_backend,
        policy=policy,
        episode_len=episode_len,
        render_cam=render_cam,
        render_dt=0.002,
    )

    ee_joint_traj = [ts.observation["qpos"] for ts in episode]
    gripper_ctrl_traj = [ts.observation["gripper_ctrl"] for ts in episode]

    is_excavator = equipment_model == "excavator_simple"
    if not is_excavator:
        if equipment_model == "vx300s_bimanual":
            for joint, ctrl in zip(ee_joint_traj, gripper_ctrl_traj):
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6] = left_ctrl
                joint[6 + 7] = right_ctrl
        else:
            for joint, ctrl in zip(ee_joint_traj, gripper_ctrl_traj):
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[6] = right_ctrl

    if fixed_excavator_box_pose is None:
        sim_backend.set_scene(
            SceneState(box_pose=np.array(episode[0].observation["env_state"], dtype=np.float64))
        )

    ee_joint_traj = np.array(ee_joint_traj)

    if is_excavator:
        replay_episode, replay_joint_traj, action_traj, follow_stats = _replay_excavator_joint_traj(
            sim_backend, ee_joint_traj
        )
        pose_eval = evaluate_excavator_pose_flow(replay_joint_traj)
        ee_pose_eval = evaluate_excavator_pose_flow(ee_joint_traj[1:])
        return action_traj, {
            "episode": replay_episode,
            "pose_eval": pose_eval,
            "ee_pose_eval": ee_pose_eval,
            "replay_joint_traj": replay_joint_traj,
            "replay_follow_stats": follow_stats,
            "ee_joint_traj": ee_joint_traj,
        }
    else:
        replay_ts = sim_backend.reset()
        replay_episode = [replay_ts]
        for action in ee_joint_traj:
            replay_ts = sim_backend.step(action)
            replay_episode.append(replay_ts)
        replay_joint_traj = np.array([ts.observation["qpos"] for ts in replay_episode[1:]])
        return ee_joint_traj, {
            "episode": replay_episode,
            "pose_eval": {},
            "ee_pose_eval": {},
            "replay_joint_traj": replay_joint_traj,
        }


def direct_sim_pipeline(
    task_name: str,
    equipment_model: str,
    episode_len: int,
    policy_cls,
    render_cam: str | None = None,
    fixed_excavator_box_pose: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    sim_backend = make_sim_backend(task_name, equipment_model=equipment_model, backend="mujoco")
    if fixed_excavator_box_pose is not None:
        sim_backend.set_scene(SceneState(box_pose=fixed_excavator_box_pose))

    policy = policy_cls(False)
    episode, actions = rollout_episode(
        backend=sim_backend,
        policy=policy,
        episode_len=episode_len,
        render_cam=render_cam,
        render_dt=0.002,
    )

    joint_traj = np.array(actions)
    pose_eval = (
        evaluate_excavator_pose_flow(joint_traj) if equipment_model == "excavator_simple" else {}
    )

    return joint_traj, {
        "episode": episode,
        "pose_eval": pose_eval,
    }

