import argparse
import os
import time
from typing import Dict

import numpy as np

from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose_for_excavator
from PACT.config.registry import StateDimRegistry
from PACT.io.hdf5_writer import write_episode_hdf5
from PACT.policies.registry import PolicyRegistry
from PACT.simulation.backends import make_sim_backend
from PACT.testbed.pipelines import direct_sim_pipeline, ee_replay_pipeline


def main(args: Dict[str, object]) -> None:
    task_name = args["task_name"]
    dataset_dir = args["dataset_dir"]
    num_episodes = args.get("num_episodes")
    if num_episodes is None:
        num_episodes = SIM_TASK_CONFIGS[task_name]["num_episodes"]
    num_episodes = int(num_episodes)
    onscreen_render = bool(args["onscreen_render"])
    equipment_model = args.get("equipment_model", "vx300s_bimanual")
    success_reward_threshold = args.get("success_reward_threshold")
    fixed_excavator_box_pose = args.get("fixed_excavator_box_pose", False)
    excavator_pipeline = args.get("excavator_pipeline", "ee_replay")
    only_save_success = bool(args.get("only_save_success", False))
    target_success_episodes = args.get("target_success_episodes")

    is_excavator = equipment_model == "excavator_simple"
    if is_excavator and excavator_pipeline not in ("ee_replay", "direct_sim"):
        raise ValueError(f"Invalid excavator_pipeline: {excavator_pipeline}")

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]
    camera_names = SIM_TASK_CONFIGS[task_name]["camera_names"]
    state_dim = StateDimRegistry.get(equipment_model)
    policy_cls = PolicyRegistry().get(task_name, equipment_model)

    render_cam_name = "angle" if onscreen_render else None
    success = []
    saved_episodes = 0

    sim_backend = make_sim_backend(task_name, equipment_model=equipment_model, backend="mujoco")
    default_success_threshold = sim_backend.reward_meta.max_reward

    for attempt_idx in range(num_episodes):
        if target_success_episodes is not None and saved_episodes >= int(target_success_episodes):
            break

        print(f"{attempt_idx=}")
        episode_success = 0
        episode_max_reward = 0.0
        episode_return = 0.0

        if is_excavator and excavator_pipeline == "direct_sim":
            print("Rollout direct joint-space scripted policy on sim backend")
            box_pose = (
                np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if fixed_excavator_box_pose
                else sample_box_pose_for_excavator()
            )
            joint_traj, info = direct_sim_pipeline(
                task_name=task_name,
                equipment_model=equipment_model,
                episode_len=episode_len,
                policy_cls=policy_cls,
                render_cam=render_cam_name,
                fixed_excavator_box_pose=box_pose,
            )
        else:
            if is_excavator:
                print("Rollout EE stage then replay on sim backend (excavator_pipeline=ee_replay)")
            print("Rollout out EE space scripted policy")
            fixed_pose = (
                np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if (is_excavator and fixed_excavator_box_pose)
                else None
            )
            joint_traj, info = ee_replay_pipeline(
                task_name=task_name,
                equipment_model=equipment_model,
                episode_len=episode_len,
                policy_cls=policy_cls,
                render_cam=render_cam_name,
                fixed_excavator_box_pose=fixed_pose,
            )

        episode = info["episode"]
        episode_return = float(np.sum([ts.reward for ts in episode[1:]]))
        episode_max_reward = float(np.max([ts.reward for ts in episode[1:]]))
        episode_success_threshold = (
            float(default_success_threshold)
            if success_reward_threshold is None
            else float(success_reward_threshold)
        )
        episode_success = int(episode_max_reward >= episode_success_threshold)

        success.append(episode_success)
        pose_eval = info.get("pose_eval") or {}
        if pose_eval:
            pr = pose_eval.get("pose_reward", 0.0)
            ee_pose_eval = info.get("ee_pose_eval") or {}
            ee_pr = ee_pose_eval.get("pose_reward")
            flags = pose_eval.get("pose_phase_flags", {})
            switch_steps = pose_eval.get("pose_phase_switch_steps", {})
            achieved = [k for k, v in flags.items() if v]
            missing = [k for k, v in flags.items() if not v]
            final_phase = max(switch_steps, key=switch_steps.get) if switch_steps else "start"
            pass_str = "PASS" if episode_success else "FAIL"
            ee_debug = (
                f" | ee_pose_reward={float(ee_pr):.2f}"
                if ee_pr is not None and abs(float(ee_pr) - float(pr)) > 1e-6
                else ""
            )
            print(
                f"{pass_str} | pose_reward={pr:.2f} | final_phase={final_phase} | "
                f"achieved={achieved} | missing={missing} | "
                f"{episode_max_reward=}, threshold={episode_success_threshold}{ee_debug}"
            )
        elif episode_success:
            print(
                f"Successful, {episode_return=}, {episode_max_reward=}, threshold={episode_success_threshold}"
            )
        else:
            print(f"Failed, {episode_max_reward=}, threshold={episode_success_threshold}")

        should_save = (not only_save_success) or bool(episode_success)
        if not should_save:
            continue

        data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/action": [],
        }
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"] = []

        joint_traj = joint_traj[:-1]
        episode = episode[:-1]

        max_timesteps = len(joint_traj)
        for t in range(max_timesteps):
            action_t = joint_traj[t]
            ts_t = episode[t]
            data_dict["/observations/qpos"].append(ts_t.observation.qpos)
            data_dict["/observations/qvel"].append(ts_t.observation.qvel)
            data_dict["/action"].append(action_t)
            for cam_name in camera_names:
                data_dict[f"/observations/images/{cam_name}"].append(
                    ts_t.observation.images[cam_name]
                )

        t0 = time.time()
        write_episode_hdf5(
            dataset_dir=dataset_dir,
            episode_idx=saved_episodes,
            data_dict=data_dict,
            state_dim=state_dim,
            attrs={
                "episode_max_reward": float(episode_max_reward),
                "episode_success_threshold": float(episode_success_threshold),
                "episode_success": int(episode_success),
            },
        )
        print(f"Saving: {time.time() - t0:.1f} secs\n")
        saved_episodes += 1

    print(f"Saved to {dataset_dir}")
    print(f"Success: {np.sum(success)} / {len(success)}")
    if only_save_success:
        print(f"Saved episodes (success only): {saved_episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--dataset_dir", action="store", type=str, help="dataset saving dir", required=True)
    parser.add_argument("--num_episodes", action="store", type=int, help="num_episodes", required=False)
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--equipment_model",
        action="store",
        type=str,
        default="vx300s_bimanual",
        help="equipment model folder under assets (e.g., vx300s_bimanual)",
    )
    parser.add_argument(
        "--success_reward_threshold",
        action="store",
        type=float,
        default=None,
        help="optional success threshold on episode max reward (default: env max reward)",
    )
    parser.add_argument(
        "--fixed_excavator_box_pose",
        action="store_true",
        help="use fixed excavator box pose for deterministic smoke test",
    )
    parser.add_argument(
        "--excavator_pipeline",
        action="store",
        type=str,
        default="ee_replay",
        choices=["ee_replay", "direct_sim"],
        help="excavator data collection pipeline: ee_replay (full flow) or direct_sim",
    )
    parser.add_argument(
        "--only_save_success",
        action="store_true",
        help="only save episodes whose episode_max_reward >= threshold",
    )
    parser.add_argument(
        "--target_success_episodes",
        action="store",
        type=int,
        default=None,
        help="stop collecting once this many successful episodes have been saved",
    )

    main(vars(parser.parse_args()))

