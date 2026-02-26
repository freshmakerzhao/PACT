import argparse
import numpy as np

from deploy_runtime.real_env import make_real_env


def main(args):
    sdk_config = {
        'fairino_robot_ip': args.fairino_robot_ip,
        'camera_source': args.camera_source,
        'camera_width': args.camera_width,
        'camera_height': args.camera_height,
    }
    env = make_real_env(
        equipment_model=args.equipment_model,
        backend=args.backend,
        camera_names=[args.camera_name],
        sdk_config=sdk_config,
    )
    ts = env.reset()
    print('Reset qpos:', ts.observation['qpos'], 'step_type=', ts.step_type.name)

    action = np.array(ts.observation['qpos'], dtype=np.float32)
    for i in range(args.steps):
        ts = env.step(action)
        print(f'step={i}, qpos={ts.observation["qpos"]}, reward={ts.reward}, step_type={ts.step_type.name}')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--equipment_model', type=str, required=True, choices=['excavator_simple', 'fairino5_single', 'fairino_fr5'])
    parser.add_argument('--backend', type=str, default='mock', choices=['mock', 'sdk'])
    parser.add_argument('--camera_name', type=str, default='top')
    parser.add_argument('--fairino_robot_ip', type=str, default='')
    parser.add_argument('--camera_source', type=str, default='0')
    parser.add_argument('--camera_width', type=int, default=640)
    parser.add_argument('--camera_height', type=int, default=480)
    parser.add_argument('--steps', type=int, default=5)
    main(parser.parse_args())
