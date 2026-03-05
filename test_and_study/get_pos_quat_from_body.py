# 获取xml中指定body的全局位姿
import mujoco
from pathlib import Path

# 相对 PACT 根目录的路径
PACT_DIR = Path(__file__).resolve().parent.parent
# 加载你的环境模型 (注意替换为你实际的 xml 路径)
model = mujoco.MjModel.from_xml_path(str(PACT_DIR / "assets/fairino5_single/single_viperx_ee_transfer_cube.xml"))
data = mujoco.MjData(model)

# 前向计算一次，不要给 qpos 赋任何值！保持最原始状态
mujoco.mj_forward(model, data)

# 获取 wrist3_link 的绝对全局位姿
wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fairino_fr5/wrist3_link')
print("--- 请把下面这两行复制到你的 XML 中 ---")
print(f"pos: {data.xpos[wrist_id]}")
print(f"quat: {data.xquat[wrist_id]}")