"""Simple Fairino deploy demo without policy.

This script drives the 6-DOF arm with a small sinusoidal joint motion
using FairinoRealEnv. It is intended for quick smoke testing of SDK
connection + servo command path.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from deploy_demo.fairino_env import FairinoRealEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    # duration: 运行时长（秒）
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    # amp: 正弦扰动幅值（弧度），默认0.05rad≈2.86°
    parser.add_argument("--amp", type=float, default=0.05, help="radians")
    # freq: 正弦扰动频率（Hz）
    parser.add_argument("--freq", type=float, default=0.2, help="Hz")
    # use_servo: 使用 ServoJ 连续伺服（更平滑）；不加则使用 MoveJ
    parser.add_argument("--use_servo", action="store_true", help="use ServoJ")
    args = parser.parse_args()

    # 6-DOF arm only, no camera/gripper
    env = FairinoRealEnv(
        robot_ip="192.168.58.2",
        camera_names=["main"],
        state_dim=6,
        joint_dof=6,
        use_radians=True,
        use_servo=args.use_servo,
    )

    ts = env.reset()
    qpos0 = np.array(ts.observation["qpos"], dtype=np.float32)

    print("[INFO] Starting simple demo. Press Ctrl+C to stop.")
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            t = time.time() - t0
            delta = args.amp * np.sin(2 * np.pi * args.freq * t)
            target = qpos0.copy()
            target[0] = qpos0[0] + delta  # move joint-1 only
            _ = env.step(target)
            time.sleep(0.008)  # ~125 Hz
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        env.disconnect()

if __name__ == "__main__":
    main()
