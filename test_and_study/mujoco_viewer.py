import mujoco
from mujoco import viewer
from pathlib import Path

# 相对 PACT 根目录的路径
PACT_DIR = Path(__file__).resolve().parent.parent
# model = mujoco.MjModel.from_xml_path(str(PACT_DIR / "assets/vx300s_single/single_viperx_ee_transfer_cube.xml"))
# model = mujoco.MjModel.from_xml_path(str(PACT_DIR / "assets/fairino5_single/single_viperx_ee_transfer_cube.xml"))
model = mujoco.MjModel.from_xml_path(str(PACT_DIR / "assets/excavator_simple/single_viperx_ee_transfer_cube.xml"))

data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)