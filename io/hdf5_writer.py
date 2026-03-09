import os
from typing import Dict

import h5py
import numpy as np


def write_episode_hdf5(
    dataset_dir: str,
    episode_idx: int,
    data_dict: Dict[str, list],
    state_dim: int,
    attrs: Dict[str, object] | None = None,
) -> None:
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    max_timesteps = len(data_dict["/action"])
    dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = True
        if attrs:
            for key, value in attrs.items():
                root.attrs[key] = value

        obs = root.create_group("observations")
        image = obs.create_group("images")
        for key in data_dict:
            if key.startswith("/observations/images/"):
                cam_name = key.split("/")[-1]
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )

        obs.create_dataset("qpos", (max_timesteps, state_dim))
        obs.create_dataset("qvel", (max_timesteps, state_dim))
        root.create_dataset("action", (max_timesteps, state_dim))

        for name, array in data_dict.items():
            root[name][...] = array

