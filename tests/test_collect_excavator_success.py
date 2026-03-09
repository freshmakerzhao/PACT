"""End-to-end regression: collect_runner with excavator ee_replay must save one success episode."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

for _p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PACT.config.registry import StateDimRegistry
from PACT.runners.collect_runner import main as collect_main


def test_collect_excavator_fixed_pose_saves_success_episode():
    """With fixed pose, ee_replay, only_save_success, target_success_episodes=1, one HDF5 has episode_success=1, episode_max_reward=4.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "excavator_e2e")
        collect_main({
            "task_name": "sim_lifting_cube_scripted",
            "dataset_dir": dataset_dir,
            "num_episodes": 5,
            "equipment_model": "excavator_simple",
            "excavator_pipeline": "ee_replay",
            "fixed_excavator_box_pose": True,
            "success_reward_threshold": 4.0,
            "only_save_success": True,
            "target_success_episodes": 1,
            "onscreen_render": False,
        })
        state_dim = StateDimRegistry.get("excavator_simple")
        path = os.path.join(dataset_dir, "episode_0.hdf5")
        assert os.path.isfile(path), f"Expected {path} to exist"
        with h5py.File(path, "r") as f:
            assert f.attrs.get("episode_max_reward") == 4.0
            assert f.attrs.get("episode_success") == 1
            assert f.attrs.get("episode_success_threshold") == 4.0
            qpos = f["/observations/qpos"][()]
            assert qpos.shape[1] == state_dim
            assert len(f["/action"][()]) == len(qpos)
