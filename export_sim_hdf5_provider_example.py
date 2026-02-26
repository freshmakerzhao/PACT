import numpy as np


# 你只需要按这个签名实现，并替换为你自己的模拟器逻辑。
def collect_episode(episode_idx: int, episode_len: int, state_dim: int, camera_names):
    qpos = np.zeros((episode_len, state_dim), dtype=np.float32)
    qvel = np.zeros((episode_len, state_dim), dtype=np.float32)
    action = np.zeros((episode_len, state_dim), dtype=np.float32)

    # ===== 示例：用你自己的模拟器替换这部分 =====
    # sim = YourSimulator(...)
    # sim.reset(seed=episode_idx)
    # for t in range(episode_len):
    #     obs = sim.get_obs()                         # 需包含关节状态
    #     act = sim.get_expert_action(obs)            # 或你的控制器动作
    #     sim.step(act)
    #     qpos[t] = obs['qpos'][:state_dim]
    #     qvel[t] = obs['qvel'][:state_dim]
    #     action[t] = act[:state_dim]
    # ==========================================

    images = {}
    for cam_name in camera_names:
        images[cam_name] = np.zeros((episode_len, 480, 640, 3), dtype=np.uint8)

    return {
        'qpos': qpos,
        'qvel': qvel,
        'action': action,
        'images': images,
    }
