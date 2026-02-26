import numpy as np


class JointSafetyFilter:
    def __init__(self, joint_min, joint_max, max_delta):
        self.joint_min = np.asarray(joint_min, dtype=np.float32)
        self.joint_max = np.asarray(joint_max, dtype=np.float32)
        self.max_delta = np.asarray(max_delta, dtype=np.float32)

    def apply(self, current_qpos, target_qpos):
        current_qpos = np.asarray(current_qpos, dtype=np.float32)
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        if np.any(~np.isfinite(target_qpos)):
            target_qpos = current_qpos.copy()
        delta = np.clip(target_qpos - current_qpos, -self.max_delta, self.max_delta)
        safe_target = current_qpos + delta
        safe_target = np.clip(safe_target, self.joint_min, self.joint_max)
        return safe_target
