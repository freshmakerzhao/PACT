import numpy as np

from .base import DeviceAdapter


class MockExcavatorAdapter(DeviceAdapter):
    def __init__(self, dt=0.02):
        self.dt = dt
        self._qpos = np.array([0.0, -0.25, -0.5, -0.5], dtype=np.float32)
        self._qvel = np.zeros(4, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def camera_names(self):
        return ['top']

    def connect(self) -> None:
        return None

    def reset(self) -> None:
        self._qpos = np.array([0.0, -0.25, -0.5, -0.5], dtype=np.float32)
        self._qvel = np.zeros(4, dtype=np.float32)

    def get_observation(self) -> dict:
        return {
            'qpos': self._qpos.copy(),
            'qvel': self._qvel.copy(),
            'images': {'top': np.zeros((480, 640, 3), dtype=np.uint8)},
            'env_state': np.zeros(1, dtype=np.float32),
        }

    def send_joint_position(self, target_qpos) -> None:
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        self._qvel = (target_qpos - self._qpos) / self.dt
        self._qpos = target_qpos

    def close(self) -> None:
        return None
