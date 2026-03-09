from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Make imports work regardless of invocation style.
# This repo mixes `import PACT.*` and legacy `from constants import ...` (constants.py lives under `PACT/`),
# so we put both the project root and `PACT/` onto sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACT_DIR = Path(__file__).resolve().parents[1]
for p in (str(_PROJECT_ROOT), str(_PACT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PACT.testbed.pipelines import ee_replay_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "tests" / "golden_excavator_replay_qpos.npy"
        ),
        help="path to save replay-stage golden qpos trajectory (.npy)",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=400,
        help="episode length (must match task config for stable comparison)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="sim_lifting_cube_scripted",
        help="task name used by pipelines/backends",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="enable onscreen render (slow, for debugging only)",
    )
    args = parser.parse_args()

    # Fixed pose matches collect_runner's fixed_excavator_box_pose branch.
    fixed_pose = np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    first_ret, info = ee_replay_pipeline(
        task_name=args.task_name,
        equipment_model="excavator_simple",
        episode_len=int(args.episode_len),
        policy_cls=__import__("PACT.policies.registry", fromlist=["PolicyRegistry"])
        .PolicyRegistry()
        .get(args.task_name, "excavator_simple"),
        render_cam="angle" if args.render else None,
        fixed_excavator_box_pose=fixed_pose,
    )
    # Excavator branch returns action_traj; ee_joint_traj is in info.
    replay_joint_traj = np.asarray(info.get("replay_joint_traj"), dtype=np.float64)
    ee_joint_traj = np.asarray(
        info.get("ee_joint_traj") if info.get("ee_joint_traj") is not None else first_ret,
        dtype=np.float64,
    )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, replay_joint_traj)
    ee_out_path = out_path.with_name("golden_excavator_ee_qpos.npy")
    np.save(ee_out_path, ee_joint_traj)

    replay_swing = replay_joint_traj[:, 0]
    replay_bucket = replay_joint_traj[:, 3]
    print(f"Saved replay golden qpos to: {out_path}")
    print(f"Saved EE golden qpos to: {ee_out_path}")
    print(f"replay qpos shape: {replay_joint_traj.shape}")
    print(f"replay swing  min/max: {float(replay_swing.min()):.4f} / {float(replay_swing.max()):.4f}")
    print(f"replay bucket min/max: {float(replay_bucket.min()):.4f} / {float(replay_bucket.max()):.4f}")

    pose_eval = info.get("pose_eval") or {}
    if pose_eval:
        print(f"replay pose_reward: {pose_eval.get('pose_reward')}")
        print(f"replay phase_flags: {pose_eval.get('pose_phase_flags')}")
        print(f"replay phase_switch_steps: {pose_eval.get('pose_phase_switch_steps')}")
        print(f"replay dump_open: {pose_eval.get('pose_phase_flags', {}).get('dump_open', False)}")
    from PACT.simulation.rewards.excavator_pose_reward import THRESH_SWING_DUMP, THRESH_BUCKET_DUMPED
    in_dump = (np.abs(replay_joint_traj[:, 0]) >= THRESH_SWING_DUMP) & (replay_joint_traj[:, 3] >= THRESH_BUCKET_DUMPED)
    if np.any(in_dump):
        replay_swing_dump = replay_joint_traj[in_dump, 0]
        replay_bucket_dump = replay_joint_traj[in_dump, 3]
        print(f"replay in dump window: swing min/max {float(replay_swing_dump.min()):.4f} / {float(replay_swing_dump.max()):.4f}, bucket min/max {float(replay_bucket_dump.min()):.4f} / {float(replay_bucket_dump.max()):.4f}")
    ee_pose_eval = info.get("ee_pose_eval") or {}
    if ee_pose_eval:
        print(f"ee pose_reward: {ee_pose_eval.get('pose_reward')}")
        print(f"ee phase_flags: {ee_pose_eval.get('pose_phase_flags')}")
        print(f"ee phase_switch_steps: {ee_pose_eval.get('pose_phase_switch_steps')}")


if __name__ == "__main__":
    main()

