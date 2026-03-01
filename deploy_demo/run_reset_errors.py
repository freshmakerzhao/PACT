"""Reset Fairino robot errors and re-enable.

Usage:
  python -m deploy_demo.run_reset_errors --robot_ip 192.168.58.2
"""

from __future__ import annotations

import argparse

import Robot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, required=True)
    args = parser.parse_args()

    robot = None
    try:
        robot = Robot.RPC(args.robot_ip)
        # reset errors
        try:
            ret = robot.ResetAllError()
            print(f"ResetAllError: {ret}")
        except Exception as exc:
            print(f"[WARN] ResetAllError failed: {exc}")

        # re-enable
        try:
            ret = robot.RobotEnable(1)
            print(f"RobotEnable: {ret}")
        except Exception as exc:
            print(f"[WARN] RobotEnable failed: {exc}")
    finally:
        if robot is not None:
            try:
                robot.CloseRPC()
            except Exception:
                pass


if __name__ == "__main__":
    main()
