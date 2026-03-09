from __future__ import annotations

from PACT import ee_sim_env as legacy_ee_sim_env


def make_ee_sim_env(task_name: str, equipment_model: str):
    return legacy_ee_sim_env.make_ee_sim_env(task_name, equipment_model)

