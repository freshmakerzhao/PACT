import collections
import numpy as np

from .factory import create_adapter
from .gateway import DeviceGateway
from .safety import JointSafetyFilter
from .timestep import StepType, TimeStep

class RealEnv:
    def __init__(self, equipment_model: str, backend: str = 'mock', camera_names=None, sdk_client=None, camera_provider=None, sdk_config=None):
        adapter = create_adapter(
            equipment_model=equipment_model,
            backend=backend,
            sdk_client=sdk_client,
            camera_provider=camera_provider,
            sdk_config=sdk_config,
        )
        self.gateway = DeviceGateway(adapter=adapter, safety_filter=self._build_safety_filter(equipment_model))
        self.camera_names = camera_names or ['top']
        self.task = type('Task', (), {'max_reward': 0})()
        self._connected = False

    def _build_safety_filter(self, equipment_model):
        if equipment_model == 'excavator_simple':
            return JointSafetyFilter(
                joint_min=[-3.14, -0.9, -0.8, -2.1],
                joint_max=[3.14, 0.3, 1.0, 0.6],
                max_delta=[0.08, 0.04, 0.04, 0.08],
            )
        return JointSafetyFilter(
            joint_min=[-2.9, -2.0, -2.9, -2.9, -2.9, -3.2, 0.0],
            joint_max=[2.9, 2.0, 2.9, 2.9, 2.9, 3.2, 1.0],
            max_delta=[0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.03],
        )

    def _format_observation(self, raw_obs):
        obs = collections.OrderedDict()
        obs['qpos'] = np.asarray(raw_obs['qpos'], dtype=np.float32)
        obs['qvel'] = np.asarray(raw_obs['qvel'], dtype=np.float32)
        obs['env_state'] = raw_obs.get('env_state', None)
        obs['images'] = {}
        for name in self.camera_names:
            if name in raw_obs['images']:
                obs['images'][name] = raw_obs['images'][name]
            else:
                obs['images'][name] = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs

    def reset(self):
        if not self._connected:
            self.gateway.connect()
            self._connected = True
        raw_obs = self.gateway.reset()
        return TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=None,
            observation=self._format_observation(raw_obs),
        )

    def step(self, action):
        raw_obs, reward = self.gateway.step(action)
        return TimeStep(
            step_type=StepType.MID,
            reward=reward,
            discount=None,
            observation=self._format_observation(raw_obs),
        )

    def close(self):
        if self._connected:
            self.gateway.close()
            self._connected = False


def make_real_env(equipment_model: str, backend: str = 'mock', camera_names=None, sdk_client=None, camera_provider=None, sdk_config=None):
    return RealEnv(
        equipment_model=equipment_model,
        backend=backend,
        camera_names=camera_names,
        sdk_client=sdk_client,
        camera_provider=camera_provider,
        sdk_config=sdk_config,
    )
