from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import Robot

from .camera_stub import make_dummy_image_dict
from .safety import JointLimits, SafetyState, clamp_to_limits, detect_large_jump, hold_position


@dataclass
class TimeStep:
    observation: dict[str, Any]
    reward: float | None = None


class FairinoRealEnv:
    # 用于评估/回放时驱动真实机械臂
    def __init__(
        self,
        robot_ip: str,
        camera_names: list[str],
        state_dim: int,
        max_jump: float = 0.2,
        joint_dof: int = 6,
        use_radians: bool = True,
        use_servo: bool = True,
        servo_period_s: float = 0.008,
        speed: float = 50.0,
        acc: float = 50.0,
        joint_limits: JointLimits | None = None,
    ):
        # --- 基础配置 ---
        # robot_ip: 控制器IP
        # camera_names: 摄像头名称列表（暂未接入相机，仅返回黑图）
        # state_dim: 观测qpos维度（双臂/单臂/挖机不同，这里单臂=6关节，可保留额外维度）
        # max_jump: 关节最大跃迁阈值（rad），超出则保持当前位置
        # joint_dof: 机械臂关节自由度（6）
        # use_radians: True 表示内部/模型使用弧度；False 表示使用角度
        # use_servo: True 使用 ServoJ 连续伺服；False 使用 MoveJ 离散命令
        # servo_period_s: 伺服周期（用于上层调度/节拍控制）
        # speed/acc: MoveJ 模式下的速度/加速度
        # joint_limits: 可传入关节软限位（单位为rad），不传则默认[-pi, pi]
        self.robot_ip = robot_ip
        self.camera_names = camera_names
        self.state_dim = state_dim
        self.max_jump = max_jump
        self.joint_dof = joint_dof
        self.use_radians = use_radians
        self.use_servo = use_servo
        self.servo_period_s = servo_period_s
        self.speed = speed
        self.acc = acc
        self._safety = SafetyState()
        self._limits: JointLimits | None = joint_limits
        self._connected = False
        self._robot: Robot | None = None
        self._servo_started = False
        self._last_send_error: int | None = None

    # --- SDK integration points ---
    def connect(self) -> None:
        """Connect to robot using Fairino SDK (fill in)."""
        # 与控制器建立连接，并做基础使能/速度设置。
        # 如果 ServoMoveStart 失败，会自动回退到 MoveJ 模式。
        if self._connected:
            return
        self._robot = Robot.RPC(self.robot_ip)
        self._connected = True
        try:
            self._robot.RobotEnable(1)
        except Exception as exc:
            print(f"[WARN] RobotEnable failed: {exc}")

        try:
            self._robot.SetSpeed(self.speed)
        except Exception as exc:
            print(f"[WARN] SetSpeed failed: {exc}")

        if self.use_servo:
            try:
                err = self._robot.ServoMoveStart()
                self._servo_started = (err == 0)
                if err != 0:
                    print(f"[WARN] ServoMoveStart error: {err}, fallback to MoveJ")
                    self.use_servo = False
            except Exception as exc:
                print(f"[WARN] ServoMoveStart failed: {exc}, fallback to MoveJ")
                self.use_servo = False

    def disconnect(self) -> None:
        """Disconnect from robot."""
        # 断开前先停止伺服，再关闭RPC连接。
        if not self._connected:
            return
        if self._robot is not None and self._servo_started:
            try:
                self._robot.ServoMoveEnd()
            except Exception as exc:
                print(f"[WARN] ServoMoveEnd failed: {exc}")
        if self._robot is not None:
            try:
                self._robot.CloseRPC()
            except Exception as exc:
                print(f"[WARN] CloseRPC failed: {exc}")
        self._connected = False
        self._servo_started = False
        self._robot = None

    def read_joint_positions(self) -> np.ndarray:
        """Read actual joint positions from SDK (fill in)."""
        # 读取关节位置：SDK可返回角度或弧度。
        # 这里统一成弧度
        if not self._connected:
            self.connect()
        if self._robot is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        # 超时/重试：避免通信卡死
        retries = 3
        last_exc: Exception | None = None
        result = None
        for attempt in range(1, retries + 1):
            try:
                if self.use_radians:
                    result = self._robot.GetActualJointPosRadian()
                else:
                    result = self._robot.GetActualJointPosDegree()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"[WARN] GetActualJointPos attempt {attempt}/{retries} failed: {exc}")
        if last_exc is not None or result is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        if isinstance(result, tuple) and len(result) == 2:
            err, joint = result
        else:
            err, joint = 0, result

        if err != 0:
            print(f"[WARN] GetActualJointPos error: {err}")
            return np.zeros(self.state_dim, dtype=np.float32)

        joint = np.array(joint[: self.joint_dof], dtype=np.float32)
        if (not self.use_radians) and joint.size > 0:
            joint = np.deg2rad(joint)

        qpos = np.zeros(self.state_dim, dtype=np.float32)
        qpos[: joint.shape[0]] = joint
        self._safety.last_qpos = qpos.copy()
        return qpos

    def read_joint_limits(self) -> JointLimits:
        """Read dynamic joint limits from SDK (fill in)."""
        # 优先从控制器读取软限位角度（单位：度），再转换为弧度
        # 返回格式为 [j1min, j1max, j2min, j2max, ...]
        if self._limits is not None:
            return self._limits
        if self._robot is not None:
            retries = 3
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    err, neg_deg = self._robot.GetJointSoftLimitDeg(0)
                    if err == 0 and len(neg_deg) >= 12:
                        lower_deg = np.array([neg_deg[i] for i in range(0, 12, 2)], dtype=np.float32)
                        upper_deg = np.array([neg_deg[i] for i in range(1, 12, 2)], dtype=np.float32)
                        if self.use_radians:
                            lower = np.deg2rad(lower_deg)
                            upper = np.deg2rad(upper_deg)
                        else:
                            lower = lower_deg
                            upper = upper_deg
                        self._limits = JointLimits(lower=lower, upper=upper)
                        return self._limits
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    print(f"[WARN] GetJointSoftLimitDeg attempt {attempt}/{retries} failed: {exc}")
            if last_exc is not None:
                print(f"[WARN] GetJointSoftLimitDeg failed after retries: {last_exc}")

        # 回退到默认[-pi, pi]
        lower = -np.ones(self.state_dim, dtype=np.float32) * np.pi
        upper = np.ones(self.state_dim, dtype=np.float32) * np.pi
        self._limits = JointLimits(lower=lower, upper=upper)
        return self._limits

    def send_joint_targets(self, target_qpos: np.ndarray) -> int | None:
        """Send joint targets to robot using SDK (fill in)."""
        # 将目标关节位置下发给控制器。
        # - use_servo=True：使用 ServoJ 连续伺服（推荐用于策略回放）
        # - use_servo=False：使用 MoveJ 离散运动（更安全但不连续）
        # 注意：SDK的 ServoJ/MoveJ 通常使用角度单位，这里会自动 rad->deg。
        if not self._connected:
            self.connect()
        if self._robot is None:
            self._last_send_error = None
            return None

        joint_cmd = np.asarray(target_qpos[: self.joint_dof], dtype=np.float32)
        if self.use_radians:
            joint_cmd = np.rad2deg(joint_cmd)

        try:
            if self.use_servo:
                exaxis_pos = [0, 0, 0, 0]
                err = self._robot.ServoJ(joint_cmd.tolist(), exaxis_pos)
                if err != 0:
                    print(f"[WARN] ServoJ error: {err}")
                self._last_send_error = err
                return err
            else:
                err = self._robot.MoveJ(joint_cmd.tolist(), vel=self.speed, acc=self.acc, tool=0, user=0)
                if err != 0:
                    print(f"[WARN] MoveJ error: {err}")
                self._last_send_error = err
                return err
        except Exception as exc:
            print(f"[WARN] send_joint_targets failed: {exc}")
            self._last_send_error = None
        return None

    # --- Env-like interface used by eval loop ---
    def reset(self) -> TimeStep:
        # reset 仅做读数与限位初始化，不做位置归零。
        if not self._connected:
            self.connect()
        qpos = self.read_joint_positions()
        self._limits = self.read_joint_limits()
        obs = {
            "qpos": qpos,
            "images": make_dummy_image_dict(self.camera_names),
        }
        return TimeStep(observation=obs, reward=None)

    def step(self, target_qpos: np.ndarray) -> TimeStep:
        # 单步执行：
        # 1) 读取当前关节
        # 2) 限位裁剪 + 跳变检测
        # 3) 下发安全目标
        # 4) 读回关节并返回观测
        current_qpos = self.read_joint_positions()
        if self._limits is None:
            self._limits = self.read_joint_limits()

        # safety clamp + jump detection
        safe_target = clamp_to_limits(target_qpos, self._limits)
        if detect_large_jump(safe_target, current_qpos, self.max_jump):
            print("[WARN] Large joint jump detected. Holding position.")
            safe_target = hold_position(current_qpos)

        self.send_joint_targets(safe_target)

        # readback
        qpos = self.read_joint_positions()
        obs = {
            "qpos": qpos,
            "images": make_dummy_image_dict(self.camera_names),
        }
        return TimeStep(observation=obs, reward=None)
