"""Simple script to move Fairino arm to a specified joint pose.

Usage example:
  python -m deploy_demo.run_movej_pose --robot_ip 192.168.58.2 --qpos 0 0 0 0 0 0 --speed 10 --acc 10
"""

from __future__ import annotations

import argparse
import time
import threading
import msvcrt
import numpy as np

from deploy_demo.fairino_env import FairinoRealEnv


ERROR_CODE_MAP = {
    23: "位姿传感器数据获取失败（检查位姿传感器网络连接/协议加载）",
    29: "Servo关节超限（检查目标关节是否超限、速度是否过大）",
}


def decode_error_code(err_tuple) -> str:
    """将 GetRobotErrorCode 返回值转为可读文本。"""
    try:
        # 常见格式: (ret, [flag, code])
        if not isinstance(err_tuple, tuple) or len(err_tuple) < 2:
            return f"未知格式: {err_tuple}"
        _, payload = err_tuple
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            code = int(payload[1])
            return ERROR_CODE_MAP.get(code, f"错误码 {code}（请对照SDK错误码表）")
        return f"未知payload: {payload}"
    except Exception as exc:
        return f"错误码解析失败: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, required=True)
    parser.add_argument("--qpos", type=float, nargs=6, required=True, help="target joints in rad")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--acc", type=float, default=10.0)
    parser.add_argument("--reset_error", action="store_true", help="reset controller errors before MoveJ")
    args = parser.parse_args()

    env = FairinoRealEnv(
        robot_ip=args.robot_ip,
        camera_names=["main"],
        state_dim=6,
        joint_dof=6,
        use_radians=True,
        use_servo=False,
        speed=args.speed,
        acc=args.acc,
    )

    robot = None
    stop_event = threading.Event()

    def stop_motion() -> None:
        if robot is None:
            return
        try:
            if hasattr(robot, "StopMove"):
                robot.StopMove()
            elif hasattr(robot, "StopMotion"):
                robot.StopMotion()
        except Exception as exc:
            print(f"[WARN] Stop motion failed: {exc}")

    def key_listener() -> None:
        # 使用按键 q 触发停止，避免 MoveJ 阻塞导致 Ctrl+C 无法及时生效
        print("[INFO] Press 'q' to stop motion immediately.")
        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"q", b"Q"):
                    stop_motion()
                    stop_event.set()
                    break
            time.sleep(0.05)
    try:
        env.reset()
        robot = getattr(env, "_robot", None)
        if robot is not None and args.reset_error:
            try:
                print(f"[INFO] ResetAllError: {robot.ResetAllError()}")
                print(f"[INFO] RobotEnable: {robot.RobotEnable(1)}")
            except Exception as exc:
                print(f"[WARN] reset/enable failed: {exc}")

        target = np.array(args.qpos, dtype=np.float32)
        target_deg = np.rad2deg(target)
        print(f"[INFO] target_qpos(rad): {target.tolist()}")
        print(f"[INFO] target_qpos(deg): {target_deg.tolist()}")

        if robot is not None:
            try:
                ret_cur, cur_deg = robot.GetActualJointPosDegree()
                print(f"[INFO] current_qpos_deg(ret={ret_cur}): {cur_deg}")
            except Exception as exc:
                print(f"[WARN] GetActualJointPosDegree failed: {exc}")

            try:
                ret_lim, lim = robot.GetJointSoftLimitDeg(0)
                if ret_lim == 0 and len(lim) >= 12:
                    lower = np.array([lim[i] for i in range(0, 12, 2)], dtype=np.float32)
                    upper = np.array([lim[i] for i in range(1, 12, 2)], dtype=np.float32)
                    inside = np.logical_and(target_deg >= lower, target_deg <= upper)
                    print(f"[INFO] soft_limit_lower_deg: {lower.tolist()}")
                    print(f"[INFO] soft_limit_upper_deg: {upper.tolist()}")
                    print(f"[INFO] target_inside_limits: {inside.tolist()}")
                else:
                    print(f"[WARN] GetJointSoftLimitDeg return: {ret_lim}, data={lim}")
            except Exception as exc:
                print(f"[WARN] GetJointSoftLimitDeg failed: {exc}")

        listener = threading.Thread(target=key_listener, daemon=True)
        listener.start()
        ret_move = env.send_joint_targets(target)
        print(f"[INFO] MoveJ/ServoJ return: {ret_move}")
        if robot is not None and ret_move not in (None, 0):
            try:
                err_code = robot.GetRobotErrorCode()
                print(f"[INFO] GetRobotErrorCode: {err_code}")
                print(f"[INFO] ErrorDetail: {decode_error_code(err_code)}")

                # 若SDK有状态包，尽量补充主/子错误码
                state_pkg = getattr(robot, "robot_state_pkg", None)
                if state_pkg is not None:
                    main_code = getattr(state_pkg, "main_code", None)
                    sub_code = getattr(state_pkg, "sub_code", None)
                    if (main_code is not None) or (sub_code is not None):
                        print(f"[INFO] robot_state_pkg main_code={main_code}, sub_code={sub_code}")
            except Exception as exc:
                print(f"[WARN] GetRobotErrorCode failed: {exc}")
        # MoveJ 返回后认为动作完成，短暂等待后自动退出
        if not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        stop_motion()
        env.disconnect()

if __name__ == "__main__":
    main()
