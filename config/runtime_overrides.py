from __future__ import annotations

from dataclasses import replace
from typing import Dict

from PACT.config.task_specs import TaskSpec


def apply_runtime_overrides(task_spec: TaskSpec, overrides: Dict[str, object]) -> TaskSpec:
    allowed = {"dataset_dir", "num_episodes", "episode_len", "camera_names"}
    filtered = {k: v for k, v in overrides.items() if k in allowed and v is not None}
    return replace(task_spec, **filtered)

