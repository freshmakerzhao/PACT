"""Replay a trained ACT policy on real Fairino arm (no gripper/camera).

Assumptions:
- Policy trained on single-arm task (sim_lifting_cube_scripted)
- Real robot has 6 joints; gripper channel (if any) is padded/ignored
- Camera input is dummy black image
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
import threading
import msvcrt

import numpy as np
import torch
from einops import rearrange

from constants import DT, SIM_TASK_CONFIGS
from deploy_demo.fairino_env import FairinoRealEnv
from policy import ACTPolicy


def get_image(ts, camera_names, device):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def make_policy(policy_config):
    return ACTPolicy(policy_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    # robot_ip: 控制器IP
    parser.add_argument("--robot_ip", type=str, default="192.168.58.2")
    # ckpt_dir: 包含 policy_best.ckpt 和 dataset_stats.pkl 的目录
    parser.add_argument("--ckpt_dir", type=str, required=True)
    # task_name: 训练时的任务名（用于读取相机名称和episode_len）
    parser.add_argument("--task_name", type=str, default="sim_lifting_cube_scripted")
    # 设备型号（用于确定state_dim）
    parser.add_argument("--equipment_model", type=str, default="fairino5_single")
    # ACT配置（需与训练一致）
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    # 运行步数
    parser.add_argument("--num_steps", type=int, default=200)
    # 策略重查询周期（步），0表示使用chunk_size
    parser.add_argument("--query_every", type=int, default=0)
    # 关节跳变阈值（rad）
    parser.add_argument("--max_jump", type=float, default=1.0)
    # 是否使用伺服
    parser.add_argument("--use_servo", action="store_true")
    # ServoJ每步最大关节变化（rad），用于避免指令速度超限
    parser.add_argument("--max_servo_step_rad", type=float, default=0.005)
    # ServoJ周期（秒），用于控制发送频率
    parser.add_argument("--servo_period_s", type=float, default=0.02)
    # 目标关节低通平滑系数，越小越平滑（0~1）
    parser.add_argument("--smoothing_alpha", type=float, default=0.2)
    # 调试打印间隔步数（避免每步打印造成卡顿）
    parser.add_argument("--print_every", type=int, default=20)
    # 无夹爪时，给策略输入的夹爪状态占位值（state_dim第7维）
    parser.add_argument("--gripper_input_value", type=float, default=0.057)
    # 发生错误时自动复位并重新使能
    parser.add_argument("--auto_reset_on_error", action="store_true")
    # 是否先移动到初始姿态
    parser.add_argument("--move_to_init", action="store_true")
    # 初始关节位置（弧度，6轴）
    parser.add_argument(
        "--init_qpos",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # 运动速度/加速度（给控制器的全局比例/参数）
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--acc", type=float, default=20.0)
    # 是否打印目标关节（调试用）
    parser.add_argument("--print_target", action="store_true")
    args = parser.parse_args()

    stop_event = threading.Event()

    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config["camera_names"]

    # 训练时单臂state_dim一般为7（含夹爪），这里做兼容：真实机器人只用6关节
    state_dim = 7
    arm_dof = 6

    policy_config = {
        "lr": 1e-4,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "equipment_model": args.equipment_model,
    }

    # load policy and stats
    ckpt_path = os.path.join(args.ckpt_dir, "policy_best.ckpt")
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = make_policy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    env = FairinoRealEnv(
        robot_ip=args.robot_ip,
        camera_names=camera_names,
        state_dim=state_dim,
        joint_dof=arm_dof,
        use_radians=True,
        use_servo=args.use_servo,
        servo_period_s=args.servo_period_s,
        max_jump=args.max_jump,
        speed=args.speed,
        acc=args.acc,
    )

    def stop_motion() -> None:
        robot = getattr(env, "_robot", None)
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
        print("[INFO] Press 'q' to stop replay immediately.")
        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"q", b"Q"):
                    stop_motion()
                    stop_event.set()
                    break
            time.sleep(0.05)

    ts = env.reset()
    if args.move_to_init:
        init_qpos = np.array(args.init_qpos, dtype=np.float32)
        prev_use_servo = env.use_servo
        env.use_servo = False
        env.send_joint_targets(init_qpos)
        time.sleep(2.0)
        env.use_servo = prev_use_servo
        # 重新读取当前状态，避免仍使用移动前的旧观测
        ts = env.reset()
    qpos = np.array(ts.observation["qpos"], dtype=np.float32)
    if qpos.shape[0] < state_dim:
        qpos = np.pad(qpos, (0, state_dim - qpos.shape[0]))
    # 无夹爪时，用固定占位值稳定策略输入分布
    if state_dim > arm_dof:
        qpos[arm_dof] = args.gripper_input_value

    print("[INFO] Starting real-robot replay. Press Ctrl+C to stop.")
    listener = threading.Thread(target=key_listener, daemon=True)
    listener.start()
    last_cmd = qpos[:arm_dof].copy()
    smoothed_target = last_cmd.copy()
    all_actions = None
    query_every = args.query_every if args.query_every > 0 else args.chunk_size
    query_every = max(1, min(query_every, args.chunk_size))
    try:
        with torch.inference_mode():
            for step_idx in range(args.num_steps):
                loop_t0 = time.perf_counter()
                if stop_event.is_set():
                    break
                qpos_norm = pre_process(qpos)
                qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                curr_image = get_image(ts, camera_names, device)

                # ACT常见部署：分块查询+逐步消费，降低抖动
                if (all_actions is None) or (step_idx % query_every == 0):
                    all_actions = policy(qpos_tensor, curr_image)
                action_idx = step_idx % query_every
                raw_action = all_actions[:, action_idx].squeeze(0).cpu().numpy()
                action = post_process(raw_action)

                # 只取前6关节下发，夹爪通道（若有）忽略
                target_qpos = action[:arm_dof]
                # 一阶低通，减少抖动
                alpha = float(np.clip(args.smoothing_alpha, 0.0, 1.0))
                smoothed_target = (1.0 - alpha) * smoothed_target + alpha * target_qpos
                target_qpos = smoothed_target

                if args.use_servo and args.max_servo_step_rad > 0:
                    lo = last_cmd - args.max_servo_step_rad
                    hi = last_cmd + args.max_servo_step_rad
                    target_qpos = np.clip(target_qpos, lo, hi)
                    last_cmd = target_qpos.copy()
                if args.print_target:
                    if step_idx % max(1, args.print_every) == 0:
                        print(f"[DEBUG] step={step_idx} target_qpos(rad): {target_qpos}")
                ts = env.step(target_qpos)

                if getattr(env, "_last_send_error", 0) not in (0, None):
                    print(f"[WARN] controller motion error: {env._last_send_error}")
                    if args.auto_reset_on_error:
                        robot = getattr(env, "_robot", None)
                        if robot is not None:
                            try:
                                print(f"[INFO] ResetAllError: {robot.ResetAllError()}")
                                print(f"[INFO] RobotEnable: {robot.RobotEnable(1)}")
                            except Exception as exc:
                                print(f"[WARN] auto reset failed: {exc}")
                    else:
                        break

                qpos = np.array(ts.observation["qpos"], dtype=np.float32)
                if qpos.shape[0] < state_dim:
                    qpos = np.pad(qpos, (0, state_dim - qpos.shape[0]))
                if state_dim > arm_dof:
                    qpos[arm_dof] = args.gripper_input_value

                # 固定频率控制环，减少时基抖动
                period = env.servo_period_s if args.use_servo else DT
                elapsed = time.perf_counter() - loop_t0
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        stop_motion()
        print("[INFO] Stopped by user.")
    finally:
        stop_event.set()
        env.disconnect()


if __name__ == "__main__":
    main()
