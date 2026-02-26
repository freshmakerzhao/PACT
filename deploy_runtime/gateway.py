import numpy as np
from constants import DT

from .ratekeeper import RateKeeper
from .safety import JointSafetyFilter


class DeviceGateway:
    def __init__(self, adapter, safety_filter: JointSafetyFilter):
        self.adapter = adapter
        self.safety_filter = safety_filter
        self.rate_keeper = RateKeeper(DT)

    def connect(self):
        self.adapter.connect()

    def reset(self):
        self.adapter.reset()
        self.rate_keeper.reset()
        return self.adapter.get_observation()

    def step(self, target_qpos):
        obs = self.adapter.get_observation()
        current_qpos = np.asarray(obs['qpos'], dtype=np.float32)
        safe_target = self.safety_filter.apply(current_qpos, target_qpos)
        self.adapter.send_joint_position(safe_target)
        self.rate_keeper.sleep_until_next()
        next_obs = self.adapter.get_observation()
        reward = self.adapter.get_reward()
        return next_obs, reward

    def close(self):
        self.adapter.close()
