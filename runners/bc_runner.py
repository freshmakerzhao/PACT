import os
import pickle
import re
from copy import deepcopy

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import compute_dict_mean, detach_dict, load_data, sample_box_pose, sample_box_pose_for_excavator, sample_insertion_pose, set_seed
from PACT.policies.learned import ACTPolicy, CNNMLPPolicy
from PACT.simulation.backends import make_sim_backend
from PACT.simulation.scene import SceneState
from visualize_episodes import save_videos


def extract_model_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format.")


def infer_start_epoch_from_ckpt_path(ckpt_path):
    ckpt_name = os.path.basename(ckpt_path)
    match = re.search(r"policy_epoch_(\d+)", ckpt_name)
    if match is None:
        return 0
    return int(match.group(1)) + 1


def build_training_checkpoint(policy, optimizer, epoch, min_val_loss, config):
    return {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "min_val_loss": float(min_val_loss),
        "config": {
            "task_name": config["task_name"],
            "seed": config["seed"],
            "policy_class": config["policy_class"],
        },
    }


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class in ("ACT", "CNNMLP"):
        return policy.configure_optimizers()
    raise NotImplementedError


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, equipment_model="vx300s_bimanual"):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    loading_status = policy.load_state_dict(extract_model_state_dict(ckpt_obj))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    if real_robot:
        from aloha_scripts.robot_utils import move_grippers
        from aloha_scripts.real_env import make_real_env

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        backend = make_sim_backend(task_name, equipment_model=equipment_model, backend="mujoco")
        env_max_reward = backend.reward_meta.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)

    num_rollouts = int(config.get("num_rollouts", 50))
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        if "sim_transfer_cube" in task_name:
            object_pose = sample_box_pose()
        elif "sim_insertion" in task_name:
            object_pose = np.concatenate(sample_insertion_pose())
        elif "sim_lifting_cube" in task_name:
            object_pose = sample_box_pose_for_excavator() if "excavator" in equipment_model else sample_box_pose()
        else:
            raise NotImplementedError
        if not real_robot:
            backend.set_scene(SceneState(box_pose=object_pose))
            ts = backend.reset()
        else:
            ts = env.reset()

        if onscreen_render:
            import matplotlib.pyplot as plt

            ax = plt.subplot()
            first_image = backend.render(height=480, width=640, camera_id=onscreen_cam) if not real_robot else ts.observation["image"]
            plt_img = ax.imshow(first_image)
            plt.ion()

        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = (
                        backend.render(height=480, width=640, camera_id=onscreen_cam)
                        if not real_robot
                        else ts.observation["image"]
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ts = env.step(target_qpos) if real_robot else backend.step(target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            import matplotlib.pyplot as plt

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, "
            f"Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(int(env_max_reward) + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config, args):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    start_epoch = 0
    resume_ckpt = args.get("resume_ckpt")
    if resume_ckpt:
        ckpt_obj = torch.load(resume_ckpt, map_location="cpu")
        loading_status = policy.load_state_dict(extract_model_state_dict(ckpt_obj))
        print(f"Resumed model from: {resume_ckpt}")
        print(loading_status)

        if isinstance(ckpt_obj, dict) and "optimizer_state_dict" in ckpt_obj:
            optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
            print("Optimizer state resumed from checkpoint.")

        if args.get("start_epoch") is not None:
            start_epoch = args["start_epoch"]
        elif isinstance(ckpt_obj, dict) and "epoch" in ckpt_obj:
            start_epoch = int(ckpt_obj["epoch"]) + 1
        else:
            start_epoch = infer_start_epoch_from_ckpt_path(resume_ckpt)

        if isinstance(ckpt_obj, dict) and "min_val_loss" in ckpt_obj:
            min_val_loss = float(ckpt_obj["min_val_loss"])

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        policy.train()
        optimizer.zero_grad()
        batch_idx = -1
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        if batch_idx < 0:
            raise ValueError("Train dataloader yielded 0 batches. Please check dataset split and paths.")
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * (epoch - start_epoch) : (batch_idx + 1) * (epoch - start_epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(build_training_checkpoint(policy, optimizer, epoch, min_val_loss, config), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        latest_ckpt_path = os.path.join(ckpt_dir, "policy_latest.ckpt")
        torch.save(build_training_checkpoint(policy, optimizer, epoch, min_val_loss, config), latest_ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(build_training_checkpoint(policy, optimizer, epoch, min_val_loss, config), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": best_epoch,
            "min_val_loss": float(min_val_loss),
            "config": {"task_name": config["task_name"], "seed": config["seed"], "policy_class": config["policy_class"]},
        },
        ckpt_path,
    )
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    import matplotlib.pyplot as plt

    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label="validation")
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")

