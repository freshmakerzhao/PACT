import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from PACT.testbed.evaluators import evaluate_excavator_pose_flow


class BasePolicy:
    def __init__(self, inject_noise: bool = False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def reset(self) -> None:
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        if self.step_count == 0:
            self.generate_trajectory(ts)

        if self.left_trajectory is not None:
            if self.left_trajectory[0]["t"] == self.step_count:
                self.curr_left_waypoint = self.left_trajectory.pop(0)
            next_left_waypoint = self.left_trajectory[0]
            left_xyz, left_quat, left_gripper = self.interpolate(
                self.curr_left_waypoint, next_left_waypoint, self.step_count
            )

        if self.right_trajectory is not None:
            if self.right_trajectory[0]["t"] == self.step_count:
                self.curr_right_waypoint = self.right_trajectory.pop(0)
            next_right_waypoint = self.right_trajectory[0]
            right_xyz, right_quat, right_gripper = self.interpolate(
                self.curr_right_waypoint, next_right_waypoint, self.step_count
            )

        if self.inject_noise:
            scale = 0.01
            if self.left_trajectory is not None:
                left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            if self.right_trajectory is not None:
                right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        if self.left_trajectory is not None:
            action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
            action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
            self.step_count += 1
            return np.concatenate([action_left, action_right])

        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
        self.step_count += 1
        return np.concatenate([action_right])


class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0},
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
        ]


class LiftingAndMovingPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30)

        tray_xyz = np.array([0.4, 0.85, 0.06])
        tray_above_xyz = tray_xyz + np.array([0, 0, 0.06])

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 50, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 70, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 100, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 140, "xyz": box_xyz + np.array([0, 0, 0.10]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 300, "xyz": tray_above_xyz, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 320, "xyz": tray_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 350, "xyz": tray_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 370, "xyz": tray_above_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
        ]


class ExcavatorMocapLiftingPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        tray_xyz = np.array([0.0, 4.0, 0.06])

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 90, "xyz": box_xyz + np.array([-0.5, 0.0, 0.30]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 140, "xyz": box_xyz + np.array([-0.5, 0.0, 0.90]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 300, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.90]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 320, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 340, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.50]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
        ]


class ExcavatorTaskSpaceDigDumpPolicy:
    def __init__(self, inject_noise: bool = False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.waypoints = []
        self.curr_waypoint = None
        self._phase = "init"
        self._last_xyz = None
        self._last_quat = None

        # 平滑参数（保守值，便于后续标定）
        self._max_xyz_step = 0.03
        self._max_quat_step_deg = 6.0

    def reset(self) -> None:
        self.step_count = 0
        self.waypoints = []
        self.curr_waypoint = None
        self._phase = "init"
        self._last_xyz = None
        self._last_quat = None

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _get_box_xyz(ts_first):
        if "box_xyz" in ts_first.observation:
            return np.array(ts_first.observation["box_xyz"], dtype=np.float64)
        return np.array(ts_first.observation["env_state"][:3], dtype=np.float64)

    @staticmethod
    def _get_dump_target_xyz(ts_first):
        if "dump_target_xyz" in ts_first.observation:
            return np.array(ts_first.observation["dump_target_xyz"], dtype=np.float64)
        return np.array([0.0, 4.0, 0.12], dtype=np.float64)

    @staticmethod
    def _desired_bucket_quat(base_quat: Quaternion, phase: str) -> Quaternion:
        if phase in ("lower_to_dig_entry", "dig_stroke"):
            return base_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-95.0)
        if phase in ("curl_and_lift", "swing_to_tray"):
            return base_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=35.0)
        if phase in ("release", "retreat"):
            return base_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-95.0)
        return base_quat

    def _limit_xyz_step(self, target_xyz):
        if self._last_xyz is None:
            return target_xyz
        delta = target_xyz - self._last_xyz
        dist = np.linalg.norm(delta)
        if dist <= self._max_xyz_step or dist < 1e-6:
            return target_xyz
        return self._last_xyz + delta * (self._max_xyz_step / dist)

    def _limit_quat_step(self, target_quat: Quaternion) -> Quaternion:
        if self._last_quat is None:
            return target_quat
        curr = Quaternion(self._last_quat)
        q_delta = target_quat * curr.conjugate
        angle = abs(q_delta.angle)
        max_angle = np.deg2rad(self._max_quat_step_deg)
        if angle <= max_angle or angle < 1e-6:
            return target_quat
        return Quaternion.slerp(curr, target_quat, amount=(max_angle / angle))

    def _build_waypoints(self, ts_first):
        box_xyz = self._get_box_xyz(ts_first)
        dump_xyz = self._get_dump_target_xyz(ts_first)
        init_mocap = ts_first.observation["mocap_pose_right"]
        start_xyz = np.array(init_mocap[:3], dtype=np.float64)
        base_quat = Quaternion(init_mocap[3:])

        dig_entry = box_xyz + np.array([1.50, 0.0, 2.00])
        dig_bottom = box_xyz + np.array([0.50, 0.0, 0.50])
        lift_xyz = box_xyz + np.array([-0.35, 0.0, 1.25])
        tray_hover = dump_xyz + np.array([0.0, 0.0, 1.00])
        take_down_xyz = dump_xyz + np.array([0.0, 0.0, 0.60])
        phases = [
            ("approach_above_box", start_xyz, base_quat, 0),
            ("lower_to_dig_entry", dig_entry, self._desired_bucket_quat(base_quat, "lower_to_dig_entry"), 50),
            ("dig_stroke", dig_bottom, self._desired_bucket_quat(base_quat, "dig_stroke"), 90),
            ("curl_and_lift", lift_xyz, self._desired_bucket_quat(base_quat, "curl_and_lift"), 140),
            ("swing_to_tray", tray_hover, self._desired_bucket_quat(base_quat, "swing_to_tray"), 230),
            ("release", tray_hover, self._desired_bucket_quat(base_quat, "release"), 350),
            ("take_down", take_down_xyz, self._desired_bucket_quat(base_quat, "release"), 370),
            ("retreat", take_down_xyz, self._desired_bucket_quat(base_quat, "retreat"), 400),
            #("return_home", start_xyz, base_quat, 400),
        ]

        self.waypoints = [
            {"t": t, "xyz": np.array(xyz, dtype=np.float64), "quat": quat.elements, "phase": phase}
            for phase, xyz, quat, t in phases
        ]

    def __call__(self, ts):
        if self.step_count == 0:
            self._build_waypoints(ts)
            self.curr_waypoint = self.waypoints.pop(0)
            self._phase = self.curr_waypoint.get("phase", "approach_above_box")
            self._last_xyz = self.curr_waypoint["xyz"].copy()
            self._last_quat = self.curr_waypoint["quat"].copy()

        if self.waypoints and self.waypoints[0]["t"] <= self.step_count:
            self.curr_waypoint = self.waypoints.pop(0)
            self._phase = self.curr_waypoint.get("phase", self._phase)

        if not self.waypoints:
            self.step_count += 1
            return np.concatenate([self._last_xyz, self._last_quat])

        next_waypoint = self.waypoints[0]
        t_curr, t_next = self.curr_waypoint["t"], next_waypoint["t"]
        if t_next <= t_curr:
            frac = 1.0
        else:
            frac = (self.step_count - t_curr) / (t_next - t_curr)
        frac = self._smoothstep(frac)

        curr_xyz = self.curr_waypoint["xyz"]
        next_xyz = next_waypoint["xyz"]
        target_xyz = curr_xyz + (next_xyz - curr_xyz) * frac

        curr_quat = Quaternion(self.curr_waypoint["quat"])
        next_quat = Quaternion(next_waypoint["quat"])
        target_quat = Quaternion.slerp(curr_quat, next_quat, amount=frac)

        target_xyz = self._limit_xyz_step(target_xyz)
        target_quat = self._limit_quat_step(target_quat)

        if self.inject_noise:
            target_xyz = target_xyz + np.random.uniform(-0.005, 0.005, size=target_xyz.shape)

        self._last_xyz = target_xyz.copy()
        self._last_quat = target_quat.elements.copy()
        self.step_count += 1
        return np.concatenate([target_xyz, target_quat.elements])


class InsertionPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        peg_info = np.array(ts_first.observation["env_state"])[:7]
        peg_xyz = peg_info[:3]

        socket_info = np.array(ts_first.observation["env_state"])[7:]
        socket_xyz = socket_info[:3]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
        ]


def test_policy(task_name, equipment_model="vx300s_bimanual"):
    onscreen_render = True
    inject_noise = False
    render_cam = "angle"

    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]
    if "sim_transfer_cube" in task_name:
        env = make_ee_sim_env("sim_transfer_cube", equipment_model=equipment_model)
        policy_cls = PickAndTransferPolicy
    elif "sim_insertion" in task_name:
        env = make_ee_sim_env("sim_insertion", equipment_model=equipment_model)
        policy_cls = InsertionPolicy
    elif "sim_lifting" in task_name:
        env = make_ee_sim_env("sim_lifting_cube", equipment_model=equipment_model)
        if equipment_model == "excavator_simple":
            policy_cls = ExcavatorTaskSpaceDigDumpPolicy
        else:
            policy_cls = LiftingAndMovingPolicy
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["images"][render_cam])
            plt.ion()

        policy = policy_cls(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation["images"][render_cam])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = float(np.max([ts.reward for ts in episode[1:]])) if episode[1:] else 0.0
        if equipment_model == "excavator_simple":
            joint_traj = np.array([ts.observation["qpos"] for ts in episode[1:]])
            pose_eval = evaluate_excavator_pose_flow(joint_traj)
            pr = pose_eval.get("pose_reward", 0.0)
            flags = pose_eval.get("pose_phase_flags", {})
            switch_steps = pose_eval.get("pose_phase_switch_steps", {})
            achieved = [k for k, v in flags.items() if v]
            missing = [k for k, v in flags.items() if not v]
            final_phase = max(switch_steps, key=switch_steps.get) if switch_steps else "start"
            pass_str = "PASS" if episode_max_reward >= 3.0 else "FAIL"
            print(
                f"{episode_idx=} {pass_str} | pose_reward={pr:.2f} | final_phase={final_phase} | "
                f"achieved={achieved} | missing={missing} | max_reward={episode_max_reward:.2f}"
            )
        elif episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

