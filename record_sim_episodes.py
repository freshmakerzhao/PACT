import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, LiftingAndMovingPolicy, ExcavatorJointSpaceDigDumpPolicy
from ee_backend import make_ee_sim_backend
from sim_backend import make_sim_backend
from utils import sample_box_pose_for_excavator


def main(args):
    """
    生成仿真演示数据（轨迹录制）。
    1) 在 ee_sim_env 中执行末端执行器(EE)空间策略，得到关节轨迹。
    2) 用夹爪控制量替换夹爪关节位置。
    3) 在 sim_env 中重放关节轨迹并记录观测。
    4) 保存为一条 episode 数据并继续下一条。
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    equipment_model = args['equipment_model'] if "equipment_model" in args else 'vx300s_bimanual'
    success_reward_threshold = args.get('success_reward_threshold', None)
    fixed_excavator_box_pose = args.get('fixed_excavator_box_pose', False)
    excavator_pipeline = args.get('excavator_pipeline', 'ee_replay')
    only_save_success = args.get('only_save_success', False)
    target_success_episodes = args.get('target_success_episodes', None)
    inject_noise = False
    render_cam_name = 'angle'
    arm_nums = 2 # 默认双臂任务，特殊型号或任务时修改
    is_excavator = equipment_model == 'excavator_simple'
    if is_excavator and excavator_pipeline not in ('ee_replay', 'direct_sim'):
        raise ValueError(f"Invalid excavator_pipeline: {excavator_pipeline}")
    use_direct_sim_for_excavator = is_excavator and excavator_pipeline == 'direct_sim'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # 从任务配置中读取 episode 长度与相机名称
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_lifting_cube_scripted':
        if equipment_model == 'excavator_simple':
            policy_cls = ExcavatorJointSpaceDigDumpPolicy
        else:
            policy_cls = LiftingAndMovingPolicy
        arm_nums = 1
    else:
        raise NotImplementedError

    if is_excavator:
        state_dim = 4
    else:
        state_dim = 14 if arm_nums == 2 else 7

    success = []
    saved_episodes = 0
    for attempt_idx in range(num_episodes):
        if target_success_episodes is not None and saved_episodes >= int(target_success_episodes):
            break

        print(f'{attempt_idx=}')
        episode_success = 0
        episode_max_reward = 0.0
        episode_return = 0.0
        episode_success_threshold = 0.0

        if use_direct_sim_for_excavator:
            print('Rollout direct joint-space scripted policy on sim backend')
            sim_backend = make_sim_backend(task_name, equipment_model=equipment_model, backend='mujoco')
            if fixed_excavator_box_pose:
                sim_backend.set_initial_object_pose(np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            else:
                sim_backend.set_initial_object_pose(sample_box_pose_for_excavator())
            ts = sim_backend.reset()
            episode_replay = [ts]
            policy = policy_cls(inject_noise)
            joint_traj = []
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = policy(ts)
                joint_traj.append(action.copy())
                ts = sim_backend.step(action)
                episode_replay.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            plt.close()
            episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
            episode_success_threshold = sim_backend.max_reward if success_reward_threshold is None else float(success_reward_threshold)
            episode_success = int(episode_max_reward >= episode_success_threshold)
        else:
            if is_excavator:
                print('Rollout EE stage then replay on sim backend (excavator_pipeline=ee_replay)')
            print('Rollout out EE space scripted policy')
            # 第一阶段：在 EE 空间执行脚本策略，得到关节轨迹
            ee_backend = make_ee_sim_backend(task_name, equipment_model=equipment_model, backend='mujoco')
            ts = ee_backend.reset()
            episode = [ts]
            policy = policy_cls(inject_noise)
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = policy(ts)
                ts = ee_backend.step(action)
                episode.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            plt.close()
            episode_return = np.sum([ts.reward for ts in episode[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode[1:]])
            if episode_max_reward == ee_backend.max_reward:
                print(f"EE stage Successful, {episode_return=}")
            else:
                print(f"EE stage Failed")

            joint_traj = [ts.observation['qpos'] for ts in episode]
            gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
            if is_excavator:
                # excavator qpos is 4D (main joints). No gripper dimension to replace.
                pass
            elif arm_nums == 2:
                for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                    joint[6] = left_ctrl
                    joint[6+7] = right_ctrl
            elif arm_nums == 1:
                for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                    joint[6] = right_ctrl
            else:
                raise NotImplementedError

            if is_excavator and fixed_excavator_box_pose:
                subtask_info = np.array([4.6, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                subtask_info = episode[0].observation['env_state'].copy()
            del ee_backend
            del episode
            del policy

            print('Replaying joint commands')
            sim_backend = make_sim_backend(task_name, equipment_model=equipment_model, backend='mujoco')
            sim_backend.set_initial_object_pose(subtask_info)
            ts = sim_backend.reset()

            episode_replay = [ts]
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(joint_traj)):
                action = joint_traj[t]
                ts = sim_backend.step(action)
                episode_replay.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)

            episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
            episode_success_threshold = sim_backend.max_reward if success_reward_threshold is None else float(success_reward_threshold)
            episode_success = int(episode_max_reward >= episode_success_threshold)

            plt.close()

        success.append(episode_success)
        if episode_success:
            print(f"Successful, {episode_return=}, {episode_max_reward=}, threshold={episode_success_threshold}")
        else:
            print(f"Failed, {episode_max_reward=}, threshold={episode_success_threshold}")

        should_save = (not only_save_success) or bool(episode_success)
        if not should_save:
            continue

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        # 组织数据：观测(qpos/qvel/图像)与动作(action)
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # 因为重放会多出 1 个动作与 1 个时间步，这里截断保持一致
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        # 按时间步对齐 action 与 observation
        for t in range(max_timesteps):
            action_t = joint_traj[t]
            ts_t = episode_replay[t]
            data_dict['/observations/qpos'].append(ts_t.observation['qpos'])
            data_dict['/observations/qvel'].append(ts_t.observation['qvel'])
            data_dict['/action'].append(action_t)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts_t.observation['images'][cam_name])

        # 写入 HDF5 数据文件
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{saved_episodes}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['episode_max_reward'] = float(episode_max_reward)
            root.attrs['episode_success_threshold'] = float(episode_success_threshold)
            root.attrs['episode_success'] = int(episode_success)
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, state_dim))
            qvel = obs.create_dataset('qvel', (max_timesteps, state_dim))
            action = root.create_dataset('action', (max_timesteps, state_dim))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        saved_episodes += 1

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')
    if only_save_success:
        print(f'Saved episodes (success only): {saved_episodes}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--equipment_model', action='store', type=str, default='vx300s_bimanual',
                        help='equipment model folder under assets (e.g., vx300s_bimanual)')
    parser.add_argument('--success_reward_threshold', action='store', type=float, default=None,
                        help='optional success threshold on episode max reward (default: env max reward)')
    parser.add_argument('--fixed_excavator_box_pose', action='store_true',
                        help='use fixed excavator box pose for deterministic smoke test')
    parser.add_argument('--excavator_pipeline', action='store', type=str, default='ee_replay',
                        choices=['ee_replay', 'direct_sim'],
                        help='excavator data collection pipeline: ee_replay (full flow) or direct_sim')
    parser.add_argument('--only_save_success', action='store_true',
                        help='only save episodes whose episode_max_reward >= threshold')
    parser.add_argument('--target_success_episodes', action='store', type=int, default=None,
                        help='stop collecting once this many successful episodes have been saved')
    
    main(vars(parser.parse_args()))

