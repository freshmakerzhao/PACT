from abc import ABC, abstractmethod


class DeviceAdapter(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def camera_names(self):
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> dict:
        raise NotImplementedError

    def get_reward(self) -> float:
        return 0.0

    @abstractmethod
    def send_joint_position(self, target_qpos) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
