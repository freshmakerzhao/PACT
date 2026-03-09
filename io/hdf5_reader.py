from __future__ import annotations

from typing import Dict

import h5py


def read_episode_hdf5(path: str) -> Dict[str, object]:
    with h5py.File(path, "r") as root:
        data = {
            "attrs": dict(root.attrs),
            "qpos": root["/observations/qpos"][()],
            "qvel": root["/observations/qvel"][()],
            "action": root["/action"][()],
            "images": {},
        }
        for cam_name in root["/observations/images"]:
            data["images"][cam_name] = root[f"/observations/images/{cam_name}"][()]
        return data

