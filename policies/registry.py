from __future__ import annotations

from typing import Callable, Dict, Tuple, Type

from PACT.policies.scripted import (
    ExcavatorTaskSpaceDigDumpPolicy,
    InsertionPolicy,
    LiftingAndMovingPolicy,
    PickAndTransferPolicy,
)


class PolicyRegistry:
    def __init__(self):
        self._registry: Dict[Tuple[str, str], Callable[[], object]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("sim_transfer_cube_scripted", "vx300s_bimanual", PickAndTransferPolicy)
        self.register("sim_insertion_scripted", "vx300s_bimanual", InsertionPolicy)
        self.register("sim_lifting_cube_scripted", "vx300s_single", LiftingAndMovingPolicy)
        self.register("sim_lifting_cube_scripted", "fairino5_single", LiftingAndMovingPolicy)
        self.register("sim_lifting_cube_scripted", "excavator_simple", ExcavatorTaskSpaceDigDumpPolicy)

    def register(self, task_name: str, equipment_model: str, policy_cls: Type) -> None:
        self._registry[(task_name, equipment_model)] = policy_cls

    def get(self, task_name: str, equipment_model: str):
        policy_cls = self._registry.get((task_name, equipment_model))
        if policy_cls is None:
            policy_cls = self._registry.get((task_name, "vx300s_bimanual"))
        if policy_cls is None:
            raise KeyError(f"No policy registered for {task_name=} {equipment_model=}")
        return policy_cls

