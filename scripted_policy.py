import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        # t是当前时间步，curr_waypoint为当前路点、next_waypoint是目标路点

        # 在两个路点之间线性插值，得到当前时刻的末端位姿与夹爪指令
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"]) # 
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # 首帧生成全轨迹，之后按时间步开环执行
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # 获取当前路点与下一个路点（左右臂独立），curr_left_waypoint 是当前路点， next_left_waypoint 是下一个关键路点
        if self.left_trajectory is not None:
            if self.left_trajectory[0]['t'] == self.step_count:
                self.curr_left_waypoint = self.left_trajectory.pop(0) # 取出当前关键路点
            next_left_waypoint = self.left_trajectory[0] # 记录下一个关键路点
            # 在路点间插值，得到当前时刻的末端位姿和夹爪开合
            left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)

        if self.right_trajectory is not None:
            if self.right_trajectory[0]['t'] == self.step_count:
                self.curr_right_waypoint = self.right_trajectory.pop(0)
            next_right_waypoint = self.right_trajectory[0]
            right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # 注入随机噪声，增加轨迹多样性
        if self.inject_noise:
            scale = 0.01
            if self.left_trajectory is not None:
                left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            if self.right_trajectory is not None:
                right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        if self.left_trajectory is not None:

            # 拼接左右臂动作（xyz + quat + gripper），直接做成1维
            action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
            action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
            self.step_count += 1
            return np.concatenate([action_left, action_right]) # 返回左右臂当前时刻的xyz、quat、gripper,8*2
        else:
            action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
            self.step_count += 1
            return np.concatenate([action_right]) # 返回右臂当前时刻的xyz、quat、gripper,8


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        # 读取初始 mocap 位姿与箱体位姿，生成搬运轨迹
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state']) # 获取箱体位姿,箱体是7维的，分别是xyz+quat。箱体在此处表示被抓取的物体
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # 右臂抓取时的目标朝向（基于初始朝向旋转）
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60) # 这里是在初始朝向基础上绕y轴旋转-60度

        # 左臂在“交接点”处的目标朝向
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90) # 绕x轴旋转90度

        # 双臂交接点（世界坐标）
        meet_xyz = np.array([0, 0.5, 0.25])

        # 下面定义了一些关键帧，也就是在t时刻机械比的位置和夹爪状态，
        # 左臂路点序列：到交接点、闭合夹爪、退回
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        # 右臂路点序列：接近箱体、抓取、移交、松开
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]

class LiftingAndMovingPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        # 读取初始 mocap 位姿与箱体位姿，生成搬运轨迹
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']

        box_info = np.array(ts_first.observation['env_state']) # 获取箱体位姿,箱体是7维的，分别是xyz+quat。箱体在此处表示被抓取的物体
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # 右臂抓取时的目标朝向（基于初始朝向旋转）
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30) # 初始朝向基础上绕x轴旋转-30度, 法奥使用
        # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60) # 初始朝向基础上绕y轴旋转-60度, vx300s使用


        tray_xyz = np.array([0.4, 0.85, 0.06])             # 盘子位置
        tray_above_xyz = tray_xyz + np.array([0, 0, 0.06]) # 盘子正上方6cm

        # 双臂交接点（世界坐标）
        meet_xyz = np.array([0, 0.5, 0.25])

        # 下面定义了一些关键帧，也就是在t时刻机械比的位置和夹爪状态，
        # 右臂路点序列：接近箱体、抓取、移交、松开
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 70, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 100, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 140, "xyz": box_xyz + np.array([0, 0, 0.10]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 300, "xyz": tray_xyz + np.array([0, 0, 0.06]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 320, "xyz": tray_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 350, "xyz": tray_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 370, "xyz": tray_xyz + np.array([0, 0, 0.06]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
        ]


class ExcavatorMocapLiftingPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        # 读取初始 mocap 位姿与箱体位姿，生成搬运轨迹（挖掘机版）
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])

        tray_xyz = np.array([0.0, 4.0, 0.06])

        # 挖掘机没有夹爪，gripper 固定为 0

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 90, "xyz": box_xyz + np.array([-0.5, 0.0, 0.30]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 140, "xyz": box_xyz + np.array([-0.5, 0.0, 0.90]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 300, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.90]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 320, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 340, "xyz": tray_xyz + np.array([-1.0, 0.0, 0.50]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
        ]


class ExcavatorDigDumpPolicy(BasePolicy):
    """Mocap-based full cycle: approach -> penetrate -> scoop -> lift -> swing -> dump -> return."""

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        base_quat = Quaternion(init_mocap_pose_right[3:])
        # Tray is on negative-y side in this scene.
        tray_xyz = np.array([0.0, -4.0, 0.06])

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 55, "xyz": box_xyz + np.array([-0.68, 0.0, 0.45]), "quat": base_quat.elements, "gripper": 0},   # approach
            {"t": 105, "xyz": box_xyz + np.array([-0.64, 0.0, 0.08]), "quat": base_quat.elements, "gripper": 0},  # penetrate deeper
            {"t": 150, "xyz": box_xyz + np.array([-0.35, 0.0, 0.92]), "quat": base_quat.elements, "gripper": 0},  # aggressive scoop + lift
            {"t": 235, "xyz": box_xyz + np.array([-0.15, 0.0, 1.28]), "quat": base_quat.elements, "gripper": 0},  # high carry
            {"t": 300, "xyz": tray_xyz + np.array([-1.55, 0.0, 1.15]), "quat": base_quat.elements, "gripper": 0}, # swing to dump side
            {"t": 330, "xyz": tray_xyz + np.array([-1.25, 0.0, 0.55]), "quat": base_quat.elements, "gripper": 0}, # lower to tray
            {"t": 350, "xyz": tray_xyz + np.array([-0.98, 0.0, 0.18]), "quat": base_quat.elements, "gripper": 0}, # deep dump
            {"t": 370, "xyz": tray_xyz + np.array([-1.03, 0.0, 0.14]), "quat": base_quat.elements, "gripper": 0}, # hold for release
            {"t": 388, "xyz": tray_xyz + np.array([-1.30, 0.0, 0.70]), "quat": base_quat.elements, "gripper": 0}, # retreat from tray
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
        ]


class ExcavatorJointSpaceDigDumpPolicy:
    """Direct joint-space cycle for sim_env rollout (skip EE->joint replay gap)."""

    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.waypoints = None
        self.curr_waypoint = None

    @staticmethod
    def _interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        return curr_waypoint["qpos"] + (next_waypoint["qpos"] - curr_waypoint["qpos"]) * t_frac

    def _build_waypoints(self, ts_first):
        box_xyz = np.array(ts_first.observation["env_state"][:3])
        dig_swing = np.clip(np.arctan2(box_xyz[1], max(box_xyz[0], 0.2)) - 0.15, -1.2, 1.2)
        dump_swing = -1.90  # command margin so realized swing can approach tray-contact zone
        start_q = np.array(ts_first.observation["qpos"]).copy()

        # Dynamic-search calibrated anchor (see test_and_study/search_dynamic_contact_pose.py):
        # q ~= [-0.41, -0.23, 0.48, -0.80] can stably trigger touch+load on fixed box.
        load_pose = np.array([dig_swing - 0.26, -0.23, 0.48, -0.80])
        secure_pose = np.array([dig_swing - 0.30, -0.30, 0.32, -0.40])
        lift_pose = np.array([dig_swing - 0.28, -0.18, 0.28, -1.05])
        carry_pose = np.array([dig_swing - 0.32, -0.20, 0.36, -1.18])

        self.waypoints = [
            {"t": 0, "qpos": start_q},
            {"t": 1, "qpos": load_pose},                                      # immediate move to dynamic-verified contact pose
            {"t": 185, "qpos": load_pose},                                    # hold for touch + load stabilization
            {"t": 235, "qpos": secure_pose},                                  # curl and secure material
            {"t": 275, "qpos": lift_pose},                                         # lift with strong bucket curl
            {"t": 305, "qpos": carry_pose},                                        # lock carry posture
            {"t": 332, "qpos": np.array([dump_swing + 0.55, -0.18, 0.40, -1.10])}, # start slow swing, keep load locked
            {"t": 354, "qpos": np.array([dump_swing + 0.28, -0.14, 0.46, -1.00])}, # transport mid
            {"t": 370, "qpos": np.array([dump_swing + 0.12, -0.08, 0.54, -0.82])}, # above tray while still curled
            {"t": 382, "qpos": np.array([dump_swing, -0.02, 0.62, -0.55])},        # descend to tray release zone
            {"t": 390, "qpos": np.array([dump_swing, -0.08, 0.72, 0.05])},         # reverse-curl release stage 1
            {"t": 396, "qpos": np.array([dump_swing + 0.08, -0.06, 0.62, 0.46])},  # reverse-curl release stage 2 + retreat
            {"t": 400, "qpos": start_q},                                      # return
        ]

    def __call__(self, ts):
        if self.step_count == 0:
            self._build_waypoints(ts)
            self.curr_waypoint = self.waypoints.pop(0)

        if self.waypoints[0]["t"] == self.step_count:
            self.curr_waypoint = self.waypoints.pop(0)
        next_waypoint = self.waypoints[0]
        action = self._interpolate(self.curr_waypoint, next_waypoint, self.step_count)

        if self.inject_noise:
            action = action + np.random.uniform(-0.01, 0.01, size=action.shape)

        self.step_count += 1
        return action

class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        # 读取 peg/socket 初始位姿，生成插入任务轨迹
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        # 左右臂抓取时的目标朝向
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        # 交接/插入参考点与右臂抬升量
        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        # 左臂路点序列：对准 socket 并参与插入
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        # 右臂路点序列：对准 peg 并参与插入
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]

def test_policy(task_name, equipment_model="vx300s_bimanual"):
    # 在 EE 环境中回放脚本策略
    onscreen_render = True
    inject_noise = False
    render_cam = 'angle'  # 可选: 'vis', 'angle', 'top'

    # 根据任务名创建对应 EE 环境
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube', equipment_model=equipment_model)
        policy_cls = PickAndTransferPolicy
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion', equipment_model=equipment_model)
        policy_cls = InsertionPolicy
    elif 'sim_lifting' in task_name:
        env = make_ee_sim_env('sim_lifting_cube', equipment_model=equipment_model)
        if equipment_model == 'excavator_simple':
            policy_cls = ExcavatorDigDumpPolicy
        else:
            policy_cls = LiftingAndMovingPolicy
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam])
            plt.ion()

        policy = policy_cls(inject_noise)
        for step in range(episode_len):
            # 逐步输出 EE 动作
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':

    test_task_name = 'sim_lifting_cube_scripted'
    equipment_model="excavator_simple"


    # test_task_name = 'sim_lifting_cube_scripted'
    # equipment_model="fairino5_single"

    # test_task_name = 'sim_lifting_cube_scripted'
    # equipment_model="vx300s_single"
    test_policy(test_task_name, equipment_model)

