from __future__ import annotations

from typing import Dict

from PACT.config.task_specs import TASK_SPECS, TaskSpec


class TaskRegistry:
    def __init__(self):
        self._registry: Dict[str, TaskSpec] = dict(TASK_SPECS)

    def get(self, task_name: str) -> TaskSpec:
        if task_name not in self._registry:
            raise KeyError(f"Unknown task: {task_name}")
        return self._registry[task_name]


class StateDimRegistry:
    _state_dims = {
        "vx300s_bimanual": 14,
        "vx300s_single": 7,
        "fairino5_single": 7,
        "excavator_simple": 4,
    }

    @classmethod
    def get(cls, equipment_model: str) -> int:
        if equipment_model not in cls._state_dims:
            raise KeyError(f"Unknown equipment model: {equipment_model}")
        return cls._state_dims[equipment_model]

