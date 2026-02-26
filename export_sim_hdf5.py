import os
import argparse
import importlib.util
from typing import Dict, Any, Callable

import h5py
import numpy as np


def _load_provider(provider_py: str, provider_fn: str) -> Callable:
    spec = importlib.util.spec_from_file_location('sim_provider_module', provider_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load provider module from: {provider_py}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, provider_fn):
        raise AttributeError(f'Provider function "{provider_fn}" not found in {provider_py}')
    fn = getattr(module, provider_fn)
    if not callable(fn):
        raise TypeError(f'{provider_fn} in {provider_py} is not callable')
    return fn


def _default_provider(episode_idx: int, episode_len: int, state_dim: int, camera_names):
    qpos = np.zeros((episode_len, state_dim), dtype=np.float32)
    qvel = np.zeros((episode_len, state_dim), dtype=np.float32)
    action = np.zeros((episode_len, state_dim), dtype=np.float32)
    images = {cam: np.zeros((episode_len, 480, 640, 3), dtype=np.uint8) for cam in camera_names}
    return {
        'qpos': qpos,
        'qvel': qvel,
        'action': action,
        'images': images,
    }


def _as_float32(name: str, value: np.ndarray, shape):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f'{name} shape mismatch: got {arr.shape}, expected {shape}')
    return arr.astype(np.float32, copy=False)


def _as_uint8_image(name: str, value: np.ndarray, shape):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f'{name} shape mismatch: got {arr.shape}, expected {shape}')
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _validate_episode_data(data: Dict[str, Any], episode_len: int, state_dim: int, camera_names):
    required = ('qpos', 'qvel', 'action', 'images')
    for key in required:
        if key not in data:
            raise KeyError(f'Missing key in provider output: {key}')

    qpos = _as_float32('qpos', data['qpos'], (episode_len, state_dim))
    qvel = _as_float32('qvel', data['qvel'], (episode_len, state_dim))
    action = _as_float32('action', data['action'], (episode_len, state_dim))

    images = data['images']
    if not isinstance(images, dict):
        raise TypeError('images must be a dict: {camera_name: np.ndarray}')

    validated_images = {}
    for cam_name in camera_names:
        if cam_name not in images:
            raise KeyError(f'Missing camera in images: {cam_name}')
        validated_images[cam_name] = _as_uint8_image(
            f'images[{cam_name}]', images[cam_name], (episode_len, 480, 640, 3)
        )

    return qpos, qvel, action, validated_images


def _write_episode_hdf5(
    dataset_dir: str,
    episode_idx: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    images: Dict[str, np.ndarray],
    sim_attr: bool,
    overwrite: bool,
):
    file_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f'File exists: {file_path}. Use --overwrite to replace.')

    with h5py.File(file_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = bool(sim_attr)

        obs_group = root.create_group('observations')
        img_group = obs_group.create_group('images')

        episode_len = qpos.shape[0]
        for cam_name, cam_data in images.items():
            img_group.create_dataset(
                cam_name,
                data=cam_data,
                shape=(episode_len, 480, 640, 3),
                dtype='uint8',
                chunks=(1, 480, 640, 3),
            )

        obs_group.create_dataset('qpos', data=qpos, shape=qpos.shape, dtype='float32')
        obs_group.create_dataset('qvel', data=qvel, shape=qvel.shape, dtype='float32')
        root.create_dataset('action', data=action, shape=action.shape, dtype='float32')


def main(args):
    os.makedirs(args.dataset_dir, exist_ok=True)

    camera_names = [cam.strip() for cam in args.camera_names.split(',') if cam.strip()]
    if len(camera_names) == 0:
        raise ValueError('camera_names must contain at least one camera.')

    if args.provider_py:
        provider = _load_provider(args.provider_py, args.provider_fn)
        print(f'Loaded provider: {args.provider_py}::{args.provider_fn}')
    else:
        provider = _default_provider
        print('Using built-in default provider (all zeros).')

    for episode_idx in range(args.num_episodes):
        data = provider(episode_idx, args.episode_len, args.state_dim, camera_names)
        qpos, qvel, action, images = _validate_episode_data(
            data=data,
            episode_len=args.episode_len,
            state_dim=args.state_dim,
            camera_names=camera_names,
        )

        _write_episode_hdf5(
            dataset_dir=args.dataset_dir,
            episode_idx=episode_idx,
            qpos=qpos,
            qvel=qvel,
            action=action,
            images=images,
            sim_attr=args.sim_attr,
            overwrite=args.overwrite,
        )
        print(f'[OK] episode_{episode_idx}.hdf5 saved')

    print('\nDone.')
    print(f'dataset_dir={args.dataset_dir}')
    print(f'num_episodes={args.num_episodes}, episode_len={args.episode_len}, state_dim={args.state_dim}')
    print(f'camera_names={camera_names}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export simulator trajectories to ACT-compatible HDF5 episodes.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Output dir for episode_*.hdf5')
    parser.add_argument('--num_episodes', type=int, required=True, help='Number of episodes to export')
    parser.add_argument('--episode_len', type=int, required=True, help='Timesteps per episode (must be fixed)')
    parser.add_argument('--state_dim', type=int, required=True, help='Action/qpos dimension (excavator=4, single-arm=7, bimanual=14)')
    parser.add_argument('--camera_names', type=str, default='top', help='Comma-separated camera names, e.g. top or top,angle')

    parser.add_argument('--provider_py', type=str, default='', help='Path to your simulator provider .py file')
    parser.add_argument('--provider_fn', type=str, default='collect_episode', help='Function name in provider file')

    parser.add_argument('--sim_attr', action='store_true', help='Set HDF5 attr sim=True (recommended for sim data)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing episode files')

    main(parser.parse_args())
