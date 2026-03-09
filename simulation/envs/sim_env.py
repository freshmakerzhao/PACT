from __future__ import annotations

from PACT import sim_env as legacy_sim_env


def make_sim_env(task_name: str, equipment_model: str):
    return legacy_sim_env.make_sim_env(task_name, equipment_model)

