from PACT.simulation.backends.mujoco import MuJoCoEESimBackend, make_ee_backend


def make_ee_sim_backend(task_name: str, equipment_model: str, backend: str = "mujoco"):
    return make_ee_backend(task_name=task_name, equipment_model=equipment_model, backend=backend)


__all__ = ["MuJoCoEESimBackend", "make_ee_sim_backend"]

