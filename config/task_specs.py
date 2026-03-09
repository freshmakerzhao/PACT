from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from constants import SIM_TASK_CONFIGS


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    dataset_dir: str
    num_episodes: int
    episode_len: int
    camera_names: list


TASK_SPECS: Dict[str, TaskSpec] = {
    name: TaskSpec(
        task_name=name,
        dataset_dir=config["dataset_dir"],
        num_episodes=config["num_episodes"],
        episode_len=config["episode_len"],
        camera_names=config["camera_names"],
    )
    for name, config in SIM_TASK_CONFIGS.items()
}


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in TASK_SPECS:
        raise KeyError(f"Unknown task spec: {task_name}")
    return TASK_SPECS[task_name]

