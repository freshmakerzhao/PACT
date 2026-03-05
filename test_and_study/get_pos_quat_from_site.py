# 获取xml中指定site的全局位姿
import mujoco
import numpy as np
from pathlib import Path

# 相对 PACT 根目录的路径
PACT_DIR = Path(__file__).resolve().parent.parent
model = mujoco.MjModel.from_xml_path(str(PACT_DIR / "assets/fairino5_single/single_viperx_ee_transfer_cube.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# 获取 TCP 锚点的 ID
tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_center')

# 获取全局位置
tcp_pos = data.site_xpos[tcp_id]

# 获取全局姿态 (将 3x3 旋转矩阵转为四元数)
tcp_quat = np.zeros(4)
mujoco.mju_mat2Quat(tcp_quat, data.site_xmat[tcp_id])

print("\n=== 请将以下数据复制到 XML 的 Mocap 中 ===")
print(f"pos: {tcp_pos.tolist()}")
print(f"quat: {tcp_quat.tolist()}")
print("==========================================\n")