from .base import DeviceAdapter


class ExcavatorSDKAdapter(DeviceAdapter):
    def __init__(self, sdk_client, camera_provider):
        self.sdk = sdk_client
        self.camera_provider = camera_provider

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def camera_names(self):
        return ['top']

    def connect(self) -> None:
        self.sdk.connect()
        self.sdk.enable_system()
        if hasattr(self.camera_provider, 'connect'):
            self.camera_provider.connect()

    def reset(self) -> None:
        self.sdk.reset_to_start_pose()

    def get_observation(self) -> dict:
        qpos, qvel = self.sdk.get_joint_state()
        image = self.camera_provider.get_top_image()
        return {
            'qpos': qpos,
            'qvel': qvel,
            'images': {'top': image},
            'env_state': None,
        }

    def send_joint_position(self, target_qpos) -> None:
        self.sdk.send_joint_target(target_qpos)

    def close(self) -> None:
        self.sdk.stop_motion()
        if hasattr(self.camera_provider, 'close'):
            self.camera_provider.close()
        self.sdk.disconnect()
