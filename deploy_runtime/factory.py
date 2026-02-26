from .adapters import (
    MockExcavatorAdapter,
    MockFairinoAdapter,
    FairinoSDKAdapter,
    ExcavatorSDKAdapter,
)
from .camera_provider import TopCameraProvider
from .sdk_clients import FairinoSDKClient, ExcavatorSDKClient


def create_adapter(equipment_model: str, backend: str = 'mock', sdk_client=None, camera_provider=None, sdk_config=None):
    sdk_config = sdk_config or {}

    if backend == 'mock':
        if equipment_model == 'excavator_simple':
            return MockExcavatorAdapter()
        if equipment_model in ('fairino5_single', 'fairino_fr5'):
            return MockFairinoAdapter()
        raise ValueError(f'Unsupported equipment_model for mock backend: {equipment_model}')

    if backend == 'sdk':
        if camera_provider is None:
            source = sdk_config.get('camera_source', 0)
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            width = int(sdk_config.get('camera_width', 640))
            height = int(sdk_config.get('camera_height', 480))
            camera_provider = TopCameraProvider(source=source, width=width, height=height)

        if sdk_client is None:
            if equipment_model in ('fairino5_single', 'fairino_fr5'):
                robot_ip = sdk_config.get('fairino_robot_ip', None)
                if not robot_ip:
                    raise ValueError('sdk backend for fairino requires fairino_robot_ip')
                sdk_client = FairinoSDKClient(robot_ip=robot_ip)
            elif equipment_model == 'excavator_simple':
                sdk_client = ExcavatorSDKClient(backend=sdk_config.get('excavator_backend', None))
            else:
                raise ValueError(f'Unsupported equipment_model for sdk backend: {equipment_model}')

        if equipment_model in ('fairino5_single', 'fairino_fr5'):
            return FairinoSDKAdapter(sdk_client=sdk_client, camera_provider=camera_provider)
        if equipment_model == 'excavator_simple':
            return ExcavatorSDKAdapter(sdk_client=sdk_client, camera_provider=camera_provider)
        raise ValueError(f'Unsupported equipment_model for sdk backend: {equipment_model}')

    raise ValueError(f'Unsupported backend: {backend}')
