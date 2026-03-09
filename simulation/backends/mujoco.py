from __future__ import annotations

from typing import Optional

import numpy as np

from PACT.core.types import Observation, RewardMeta, StepResult
from PACT.simulation.backends.base import SimulationBackend
from PACT.simulation.scene.scene_state import SceneState
from PACT.simulation.envs.sim_env import make_sim_env
from PACT.simulation.envs.ee_sim_env import make_ee_sim_env


class MuJoCoSimBackend(SimulationBackend):
    def __init__(self, task_name: str, equipment_model: str):
        self._env = make_sim_env(task_name, equipment_model)
        self._last_ts = None

    def reset(self) -> StepResult:
        self._last_ts = self._env.reset()
        return self._wrap_ts(self._last_ts)

    def step(self, action: np.ndarray) -> StepResult:
        self._last_ts = self._env.step(action)
        return self._wrap_ts(self._last_ts)

    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        return self._env._physics.render(height=height, width=width, camera_id=camera_id)

    def get_obs(self) -> Observation:
        if self._last_ts is None:
            raise RuntimeError("Backend has no observation. Call reset() first.")
        return self._wrap_ts(self._last_ts).observation

    @property
    def reward_meta(self) -> RewardMeta:
        return RewardMeta(max_reward=float(self._env.task.max_reward))

    @property
    def max_reward(self) -> float:
        return float(self._env.task.max_reward)

    def set_scene(self, scene_state: Optional[object]) -> None:
        if scene_state is None:
            return
        if not isinstance(scene_state, SceneState):
            raise TypeError("scene_state must be a SceneState")
        if scene_state.box_pose is not None:
            from PACT import sim_env as legacy_sim_env
            from PACT import ee_sim_env as legacy_ee_sim_env

            legacy_sim_env.BOX_POSE[0] = scene_state.box_pose
            legacy_ee_sim_env.EXCAVATOR_BOX_POSE[0] = scene_state.box_pose

    def set_initial_object_pose(self, object_pose: np.ndarray) -> None:
        self.set_scene(SceneState(box_pose=object_pose))

    @staticmethod
    def _wrap_ts(ts) -> StepResult:
        extras = {
            key: value
            for key, value in ts.observation.items()
            if key not in ("qpos", "qvel", "env_state", "images")
        }
        obs = Observation(
            qpos=np.array(ts.observation["qpos"]),
            qvel=np.array(ts.observation["qvel"]),
            env_state=np.array(ts.observation["env_state"]),
            images=dict(ts.observation.get("images", {})),
            extras=extras,
        )
        reward = float(ts.reward) if ts.reward is not None else 0.0
        return StepResult(observation=obs, reward=reward, done=bool(ts.last()))


class MuJoCoEESimBackend(SimulationBackend):
    def __init__(self, task_name: str, equipment_model: str):
        self._env = make_ee_sim_env(task_name, equipment_model)
        self._last_ts = None

    def reset(self) -> StepResult:
        self._last_ts = self._env.reset()
        return self._wrap_ts(self._last_ts)

    def step(self, action: np.ndarray) -> StepResult:
        self._last_ts = self._env.step(action)
        return self._wrap_ts(self._last_ts)

    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        return self._env._physics.render(height=height, width=width, camera_id=camera_id)

    def get_obs(self) -> Observation:
        if self._last_ts is None:
            raise RuntimeError("Backend has no observation. Call reset() first.")
        return self._wrap_ts(self._last_ts).observation

    @property
    def reward_meta(self) -> RewardMeta:
        return RewardMeta(max_reward=float(self._env.task.max_reward))

    @property
    def max_reward(self) -> float:
        return float(self._env.task.max_reward)

    def set_scene(self, scene_state: Optional[object]) -> None:
        if scene_state is None:
            return
        if not isinstance(scene_state, SceneState):
            raise TypeError("scene_state must be a SceneState")
        if scene_state.box_pose is not None:
            from PACT import sim_env as legacy_sim_env
            from PACT import ee_sim_env as legacy_ee_sim_env

            legacy_sim_env.BOX_POSE[0] = scene_state.box_pose
            legacy_ee_sim_env.EXCAVATOR_BOX_POSE[0] = scene_state.box_pose

    @staticmethod
    def _wrap_ts(ts) -> StepResult:
        extras = {
            key: value
            for key, value in ts.observation.items()
            if key not in ("qpos", "qvel", "env_state", "images")
        }
        obs = Observation(
            qpos=np.array(ts.observation["qpos"]),
            qvel=np.array(ts.observation["qvel"]),
            env_state=np.array(ts.observation["env_state"]),
            images=dict(ts.observation.get("images", {})),
            extras=extras,
        )
        reward = float(ts.reward) if ts.reward is not None else 0.0
        return StepResult(observation=obs, reward=reward, done=bool(ts.last()))


def make_sim_backend(task_name: str, equipment_model: str, backend: str = "mujoco") -> SimulationBackend:
    if backend == "mujoco":
        return MuJoCoSimBackend(task_name=task_name, equipment_model=equipment_model)
    raise NotImplementedError(f"Unsupported sim backend: {backend}")


def make_ee_backend(task_name: str, equipment_model: str, backend: str = "mujoco") -> SimulationBackend:
    if backend == "mujoco":
        return MuJoCoEESimBackend(task_name=task_name, equipment_model=equipment_model)
    raise NotImplementedError(f"Unsupported ee sim backend: {backend}")

