import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            qpos_traj = root['/observations/qpos'][()]
            env_state_traj = root['/observations/env_state'][()]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = qpos_traj[start_ts]
            qvel = root['/observations/qvel'][start_ts]
            env_xyz = env_state_traj[start_ts]
            qpos_10d = np.concatenate([qpos, env_xyz])

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                qpos_for_action = qpos_traj[start_ts:]
                action_len = episode_len - start_ts
            else:
                action_start = max(0, start_ts - 1)
                action = root['/action'][action_start:] # hack, to make timesteps more aligned
                qpos_for_action = qpos_traj[action_start:]
                action_len = episode_len - action_start # hack, to make timesteps more aligned

        delta_action = action.copy()
        delta_action[:, :6] = action[:, :6] - qpos[:6] 

        delta_action_10d = np.zeros((action_len, 10), dtype=np.float32)
        delta_action_10d[:, :7] = delta_action

        self.is_sim = is_sim
        padded_action = np.zeros((original_action_shape[0], 10), dtype=np.float32)
        padded_action[:action_len] = delta_action_10d
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos_10d).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["delta_action_mean"]) / self.norm_stats["delta_action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    all_delta_action_data = []
    
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            env_state = root['/observations/env_state'][()] # 读xyz

            episode_len = action.shape[0]
            # ===== 拼接成 10 维并存入统计列表 =====
            qpos_10d = np.concatenate([qpos, env_state], axis=1)
            action_10d = np.zeros((episode_len, 10), dtype=np.float32)
            action_10d[:, :7] = action

            all_qpos_data.append(torch.from_numpy(qpos_10d))
            all_action_data.append(torch.from_numpy(action_10d))
            
            # 遍历每个起点 t，计算它与其后所有 action 的差
            for t in range(episode_len):
                delta_chunk = action[t:].copy()
                delta_chunk_10d = np.zeros((len(delta_chunk), 10), dtype=np.float32)
                delta_chunk_10d[:, :6] = delta_chunk[:, :6] - qpos[t, :6]
                delta_chunk_10d[:, 6] = delta_chunk[:, 6]
                all_delta_action_data.append(torch.from_numpy(delta_chunk_10d))

    # 改用 cat 展平所有数据，因为每个 delta_chunk 长度不一致
    all_qpos_data = torch.cat(all_qpos_data, dim=0).float()
    all_action_data = torch.cat(all_action_data, dim=0).float()
    all_delta_action_data = torch.cat(all_delta_action_data, dim=0).float()

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # normalize residual action data
    delta_action_mean = all_delta_action_data.mean(dim=0, keepdim=True)
    delta_action_std = all_delta_action_data.std(dim=0, keepdim=True)
    delta_action_std = torch.clip(delta_action_std, 1e-2, np.inf)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "delta_action_mean": delta_action_mean.numpy().squeeze(), "delta_action_std": delta_action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [-0.1, 0.3]
    y_range = [0.3, 0.7]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


        # <!-- ============  x=[-0.2, 0.0], y=[0.4, 0.6] ============ -->
def sample_box_pose_eval():
    # x_range = [0.0, 0.2]
    # y_range = [0.4, 0.6]
    x_range = [-0.1, 0.3]
    y_range = [0.3, 0.7]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_box_pose_eval_ring():
    outer_x_range = [-0.1, 0.3]
    outer_y_range = [0.3, 0.7]
    inner_x_range = [0.0, 0.2]
    inner_y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    outer_ranges = np.vstack([outer_x_range, outer_y_range, z_range])

    for _ in range(1000):
        cube_position = np.random.uniform(outer_ranges[:, 0], outer_ranges[:, 1])
        x, y = cube_position[0], cube_position[1]
        in_inner = (inner_x_range[0] <= x <= inner_x_range[1]) and (inner_y_range[0] <= y <= inner_y_range[1])
        if not in_inner:
            cube_quat = np.array([1, 0, 0, 0])
            return np.concatenate([cube_position, cube_quat])

    raise RuntimeError('Failed to sample box pose in eval ring after 1000 attempts')

def sample_box_pose_for_excavator():
    x_range = [3.4, 4.4]
    y_range = [-1, 1]
    z_range = [0.25, 0.25]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
