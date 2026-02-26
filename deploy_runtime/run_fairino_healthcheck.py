import argparse
import time
import numpy as np

from deploy_runtime.sdk_clients import FairinoSDKClient


def main(args):
    client = FairinoSDKClient(robot_ip=args.robot_ip)

    print('[1/5] connect')
    client.connect()

    print('[2/5] enable robot')
    client.enable_robot()

    print('[3/5] read joint state')
    qpos, qvel = client.get_joint_state()
    print('qpos:', qpos)
    print('qvel:', qvel)

    print('[4/5] send one safe hold command')
    target = np.array(qpos, dtype=np.float32)
    client.servo_j(target)
    time.sleep(0.05)

    print('[5/5] stop and disconnect')
    client.stop()
    client.disconnect()

    print('Healthcheck done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_ip', type=str, required=True, help='Fairino robot controller IP')
    main(parser.parse_args())
