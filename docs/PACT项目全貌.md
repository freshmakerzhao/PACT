# PACT 项目全貌文档

> **维护规则**：每次对 `PACT/` 下源码的修改（新增文件、删除文件、修改接口签名、变更数据流）都 **必须** 同步更新本文档对应章节。此规则已写入 `.cursor/rules/doc-sync.mdc`。

最后更新：2026-03-09

---

## 1. 项目定位

PACT（Policy ACT）是一个**基于 ACT（Action Chunking with Transformers）模仿学习框架**的仿真实验平台，当前目标是工程机械（挖掘机）的智能控制。项目从 ALOHA 双臂机器人代码库演化而来，已扩展支持单臂机器人（vx300s / fairino5）和简化挖掘机（excavator_simple）。

核心能力：**数据采集（脚本策略演示）→ HDF5 录制 → ACT 模型训练 → 仿真回放评测 → 视频可视化**，已跑通完整闭环。

---

## 2. 目录结构

```
PACT/
├── assets/                          # MuJoCo MJCF 模型文件
│   ├── excavator_simple/            # 4-DOF 简化挖掘机
│   │   ├── single_viperx_transfer_cube.xml      # 关节空间环境场景
│   │   ├── single_viperx_ee_transfer_cube.xml   # EE 空间环境场景
│   │   ├── mjmodel_act_simple.xml               # 挖掘机本体模型
│   │   └── scene.xml                            # 通用场景（地面、灯光等）
│   ├── vx300s_bimanual/             # 双臂 ViperX 300s
│   ├── vx300s_single/               # 单臂 ViperX 300s
│   ├── fairino5_single/             # 法奥 FR5 单臂
│   └── fairino5_single_修改模型角度难部署/  # FR5 历史调试版本
│
├── detr/                            # ACT 模型架构（来自 DETR 改造）
│   ├── main.py                      # 模型 & 优化器构建入口
│   ├── models/
│   │   ├── __init__.py              # 导出 build_ACT_model / build_CNNMLP_model
│   │   ├── detr_vae.py              # ★ 核心：DETRVAE (CVAE) 模型定义
│   │   ├── backbone.py              # ResNet18/50 backbone + 位置编码
│   │   ├── transformer.py           # Transformer Encoder/Decoder
│   │   └── position_encoding.py     # Sine/Learned 位置编码
│   └── setup.py
│
├── core/                             # 类型与接口层（types / interfaces）
├── simulation/                       # 统一后端 + env 包装 + 场景/奖励
│   ├── backends/                     # MuJoCo backend 统一接口
│   ├── envs/                         # sim_env / ee_sim_env 包装
│   ├── scene/                        # SceneState（替代 BOX_POSE 直连）
│   └── rewards/                      # 挖机姿态流程评分（pose-only）
├── policies/                         # 策略分层（scripted / learned / registry）
├── testbed/                          # rollout / pipelines / evaluators / render
├── io/                               # HDF5 读写、stats、manifest
├── config/                           # task_specs / registry / overrides
├── runners/                          # collect/train/eval runner
├── tests/                            # 回归：golden reward 检查、ee_replay pipeline、collect e2e
├── tools/                            # generate_excavator_reward_golden 等
├── constants.py          # ★ 全局常量：DT、关节名、初始位姿、夹爪归一化函数、任务配置
├── sim_env.py            # ★ 关节空间仿真环境（make_sim_env + Task 类）
├── ee_sim_env.py         # ★ 末端执行器(EE)空间仿真环境（make_ee_sim_env + EETask 类）
├── scripted_policy.py    # ★ 兼容入口：转调 policies/scripted
├── ee_backend.py         # ★ 兼容入口：转调 simulation/backends
├── sim_backend.py        # ★ 兼容入口：转调 simulation/backends
├── record_sim_episodes.py# ★ 兼容入口：转调 runners/collect_runner
├── policy.py             # ★ 兼容入口：转调 policies/learned
├── imitate_episodes.py   # ★ 兼容入口：转调 runners/train_runner/eval_runner
├── utils.py              # 数据加载(EpisodicDataset)、归一化统计、随机采样、工具函数
├── visualize_episodes.py # 可视化：HDF5→视频、关节轨迹绘图
├── 常用命令.txt            # CLI 命令速查
├── requirements.txt
├── conda_env.yaml
└── data_sim_episodes/    # 生成的 HDF5 数据目录（可通过 PACT_DATA_DIR 环境变量覆盖）
```

**已清理（fs/decoupling）**：`study_*.py`、`trajectories.py`、`test_and_study/` 等仅用于调试的脚本与 sim_env 内硬件遥操作死代码已移除，主流程仅依赖上述模块。

---

## 3. 核心数据流

### 3.1 数据采集流（离线）

```
ScriptedPolicy（挖机为分阶段挖掘轨迹策略）
       │
       ▼
EESimBackend（当前 MuJoCoEESimBackend）
  ├─ reset/step
  └─ 产出 observation: qpos/qvel/images/mocap_pose/gripper_ctrl/env_state
       │
       ▼
ee_sim_env（EE 空间任务定义）
  ├─ mocap 控制末端位姿
  ├─ MuJoCo IK 解算关节
  └─ 输出底层仿真状态
       │
       │  提取 joint_traj = [ts.observation['qpos'] for ts in episode]
       │  替换夹爪：joint[6] = normalize(ctrl[0])（非挖掘机时）
      ▼
SimBackend（当前 MuJoCoSimBackend）
  ├─ set_initial_object_pose(...)
  ├─ reset()
  ├─ step(joint_traj[t])
  └─ 录制 observation → HDF5
       │
       ▼
HDF5 文件: episode_N.hdf5
  /observations/qpos          (T, state_dim)     float64
  /observations/qvel          (T, state_dim)     float64
  /observations/images/{cam}  (T, 480, 640, 3)   uint8
  /action                     (T, state_dim)     float64
  root.attrs['sim'] = True
```

> 模块化入口：`runners/collect_runner.py` → `testbed/pipelines.py` → `io/hdf5_writer.py`。

### 3.2 训练流

```
HDF5 episodes
    │
    ▼
EpisodicDataset（utils.py）
  ├─ 随机采样 start_ts
  ├─ 取 qpos[start_ts] + images[start_ts] 作为观测
  ├─ 取 action[start_ts:] 作为目标序列（padding + is_pad mask）
  ├─ 归一化：(x - mean) / std
  └─ 输出 (image_data, qpos_data, action_data, is_pad)
    │
    ▼
ACTPolicy.__call__(qpos, image, actions, is_pad)    # 训练模式
  ├─ image normalize (ImageNet mean/std)
  ├─ DETRVAE.forward(qpos, image, None, actions, is_pad)
  │   ├─ Encoder: [CLS] + qpos_embed + action_embed → latent μ, logσ²
  │   ├─ Reparametrize → z (32-dim)
  │   └─ Decoder: image features(ResNet18) + qpos + z → action_chunk
  └─ Loss = L1(actions, a_hat) + kl_weight × KL(μ, logσ²)
    │
    ▼
Checkpoint: policy_best.ckpt / policy_epoch_N_seed_S.ckpt / policy_latest.ckpt
```

### 3.3 评测/推理流

```
SimBackend.reset()
    │
    ▼
循环 max_timesteps 步:
  obs = ts.observation
  qpos = pre_process(obs['qpos'])        # (qpos - mean) / std
  image = get_image(ts, camera_names)    # (B,num_cam,3,H,W) float cuda
    │
    ▼
  ACTPolicy.__call__(qpos, image)         # 推理模式：z = 0 向量
    → action_chunk (1, chunk_size, state_dim)
    │
    │ temporal_agg 时：指数加权融合历史 chunk
    │ 非 temporal_agg 时：按 query_frequency 截取当前步动作
    ▼
  action = post_process(raw_action)       # action * std + mean
  ts = backend.step(action)
    │
    ▼
计算 success_rate / avg_return，保存视频
```

---

## 4. 关键模块详解

### 4.0 模块化分层（新增）

- `core/`：`types.py`/`interfaces.py` 统一数据类型与后端/策略接口。
- `simulation/backends/`：统一后端接口，MuJoCo 实现集中在 `mujoco.py`。
- `policies/`：`scripted/` 迁入脚本策略，`learned/` 迁入学习策略，`registry.py` 负责分派。
- `testbed/`：`rollout.py` 执行单回合；`pipelines.py` 负责 `ee_replay`/`direct_sim`，在 `info["pose_eval"]` 中输出 `pose_reward`/`pose_phases`/`pose_phase_flags`/`pose_phase_switch_steps`。对 `ee_replay`，`pose_eval` 统一基于 **replay 后 sim 真实 qpos** 计算，以与环境 reward / HDF5 观测一致；**挖掘机**（`excavator_simple`）使用专用 **replay follower**（每目标多步跟踪 + 卸料窗口 hold + 最终 dump hold），不再单步直推 qpos，以保证 replay 达到 dump_open 4.0。同时额外提供 `info["ee_pose_eval"]`、`info["replay_follow_stats"]` 作为调试信息。`evaluators.py` 调用 `ExcavatorPoseReward` 做 pose-only 评分（无硬编码阈值，阈值集中在 reward 模块）。
- `io/`：HDF5 写入与 stats 保存集中管理。
- `runners/`：`collect_runner`/`train_runner`/`eval_runner` 作为新入口。
- 旧脚本（`record_sim_episodes.py`/`imitate_episodes.py`/`scripted_policy.py`/`policy.py`）保留为壳层以兼容旧命令。

### 4.1 constants.py — 全局配置中心

| 常量 | 值 | 说明 |
|------|-----|------|
| `DT` | 0.02 | 控制时间步（50Hz） |
| `EXCAVATOR_MAIN_JOINTS` | `('j1_swing','j2_boom','j3_stick','j4_bucket')` | 挖掘机 4 个主关节名 |
| `EXCAVATOR_START_POSE` | `[0.0, -0.25, -0.5, -0.5]` | 挖掘机初始关节角度 |
| `SIM_TASK_CONFIGS` | dict | 任务配置：数据目录、episode 数量/长度、相机名 |
| `XML_DIR` | `PACT/assets/` 的绝对路径 | MJCF 模型根目录 |
| 夹爪归一化函数 | lambda | `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` 等，将原始夹爪位置映射到 [0,1] |

**任务配置表**（`SIM_TASK_CONFIGS`）：

| task_name | episode_len | camera_names | 说明 |
|-----------|-------------|--------------|------|
| `sim_transfer_cube_scripted` | 400 | `['top']` | 双臂搬方块 |
| `sim_insertion_scripted` | 400 | `['top']` | 双臂插钉 |
| `sim_lifting_cube_scripted` | 400 | `['top']` | 单臂/挖掘机搬方块 |

### 4.2 sim_env.py — 关节空间环境

**入口函数**：`make_sim_env(task_name, equipment_model) -> dm_control Environment`

根据 `task_name` 和 `equipment_model` 加载不同 MJCF + Task：

| task_name 包含 | equipment_model | Task 类 | state_dim |
|----------------|-----------------|---------|-----------|
| `sim_transfer_cube` | `vx300s_bimanual` | `TransferCubeTask` | 14 |
| `sim_insertion` | `vx300s_bimanual` | `InsertionTask` | 14 |
| `sim_lifting_cube` | `excavator_simple` | `ExcavatorSimpleLiftingCubeTask` | 4 |
| `sim_lifting_cube` | 其他 | `LiftingCubeTask` | 7 |

**Task 类层次**：
- `BimanualViperXTask(base.Task)` — 双臂/单臂夹爪设备基类
  - `before_step(action, physics)` — 将归一化 action 映射为 MuJoCo ctrl
  - `get_qpos/get_qvel/get_observation/get_reward` — 读取 `physics.data.*`
- `ExcavatorSimpleLiftingCubeTask(base.Task)` — 挖掘机专用，不继承 BimanualViperX
  - `before_step` 直接 `np.copyto(physics.data.ctrl, action)`
  - `get_qpos/get_qvel` 按关节名索引 `physics.data.qpos/qvel`
  - `get_reward` 使用 pose-only 评分（`ExcavatorPoseReward.step_reward`），不依赖刚体接触

**观测输出结构**（所有 Task 共用）：
```python
obs = {
    'qpos': ndarray(state_dim,),       # 归一化关节位置
    'qvel': ndarray(state_dim,),       # 归一化关节速度
    'env_state': ndarray(...),         # 环境物体状态（box 位姿等）
    'images': {
        'top':   ndarray(480,640,3),   # uint8
        'angle': ndarray(480,640,3),
        'vis':   ndarray(480,640,3),
    }
}
```

**挖掘机额外观测（task-space 调试）**：
- `box_xyz`：方块位置（与 env_state 对齐）
- `bucket_tip_xyz`：铲斗尖端 site 世界坐标
- `dump_target_xyz`：卸料目标 site 世界坐标

**全局变量 `BOX_POSE`**：由外部在 reset 前设置方块初始位姿，`initialize_episode` 中写入 `physics.data.qpos`。

### 4.3 ee_sim_env.py — EE 空间环境

**入口函数**：`make_ee_sim_env(task_name, equipment_model) -> dm_control Environment`

与 `sim_env` 的区别：动作空间为**末端位姿 (xyz+quat) + 夹爪**，MuJoCo 通过 mocap body 做 IK。

**Task 类层次**：
- `BimanualViperXEETask(base.Task)` — EE 空间基类
  - `before_step` 写入 `physics.data.mocap_pos/mocap_quat`
  - `initialize_robots(physics)` — 设置初始关节位姿 + mocap 位姿
  - 挖掘机分支：当 action 长度 == 4 时直接写 ctrl（关节空间），否则走 mocap
  - 额外输出 `obs['mocap_pose_right']` 和 `obs['gripper_ctrl']`
- `ExcavatorSimpleLiftingCubeEETask` — 挖掘机 EE 版搬方块，奖励使用 `ExcavatorPoseReward`（与 SIM 共用，pose-only）
  - 支持外部固定 box pose（由 backend 写入共享容器）
  - 额外输出 `box_xyz`/`bucket_tip_xyz`/`dump_target_xyz`，用于 task-space 策略

### 4.4 scripted_policy.py — 脚本策略（壳层）

- 策略实现已迁至 `policies/scripted/`，旧文件仅做兼容导出。

**BasePolicy**：
- 首帧调用 `generate_trajectory(ts_first)` 生成关键路点序列
- 后续每步在相邻路点间线性插值（位置 + 四元数 + 夹爪）
- 可选噪声注入（`inject_noise`，scale=0.01）

| 策略类 | 适用 task | 说明 |
|--------|-----------|------|
| `PickAndTransferPolicy` | sim_transfer_cube | 双臂：右臂抓→移交→左臂接 |
| `InsertionPolicy` | sim_insertion | 双臂：抓 peg+socket → 插入 |
| `LiftingAndMovingPolicy` | sim_lifting_cube (非挖掘机) | 单臂：抓→抬→移至托盘→放 |
| `ExcavatorDigDumpPolicy` | sim_lifting_cube (挖掘机, EE) | 挖机完整循环：接近→切入→收斗→抬升→回转→卸料→回位 |
| `ExcavatorTaskSpaceDigDumpPolicy` | sim_lifting_cube (挖掘机, EE) | 任务空间阶段规划：mocap 目标→仿真解出关节→回放 |
| `ExcavatorJointSpaceDigDumpPolicy` | sim_lifting_cube (挖掘机, SIM) | 关节空间回放策略，保留作对照/回滚 |

**挖机策略执行要点**（`ExcavatorTaskSpaceDigDumpPolicy` + `ExcavatorJointSpaceDigDumpPolicy`）：
- 使用 task-space 关键点（box/tray/site）规划完整作业循环：接近、切入、收斗、抬升、回转、下放、回位。
- 在 `ee_sim_env` 执行后，统一提取关节轨迹并在 `sim_env` 重放，保持训练数据形态不变（state_dim=4）。
- 卸料段显式包含“回转→下放→释放”，避免仅在固定 swing 角开斗导致落点误差。
- task-space 默认参数会在 swing 阶段抬高并收斗，确保能越过托盘高度后再下放。
- 关节空间策略保留为对照/回滚路线。

**`ExcavatorJointSpaceDigDumpPolicy` 控制流改进**（2026-03-06）：

1. **距离自适应关节角**：boom/stick/bucket 不再硬编码，而是根据 box 径向距离 r 在近距锚点（r=3.4）和远距锚点（r=4.6）之间线性插值。锚点值以类常量 `_LOAD_NEAR/_LOAD_FAR` 等形式存储，方便后续标定调整。
2. **平滑接近轨迹**：将原先 t=0→t=1 的瞬时跳变改为 80 步（1.6 秒）的三段平滑接近（start → approach → load），消除大惯量关节的振荡和不可控碰撞。
3. **运输阶段铲斗锁定**：回转运输阶段（t=305~370）bucket 角度锁定在 carry 阶段的强卷入值，防止离心力导致载荷掉落。
4. **半闭环状态反馈**：`__call__` 在 `hold_for_load` 阶段读取 `ts.observation["env_state"]` 中的 box_z，若 box 已被抬起（z > 0.30）则调用 `_advance_to_phase("secure")` 提前进入下一阶段，避免无意义等待。
5. **卸料缓释**：反卷释放从原先 14 步扩展到 ~28 步（t=370~398），分 3 阶段线性展开 bucket，防止载荷被甩出托盘。
6. **Bug 修复**：`ExcavatorDigDumpPolicy` 中 `tray_xyz` 的 y 坐标从 `-4.0` 修正为 `+4.0`，与场景 XML 中的托盘位置一致。

**铲斗动作与步骤顺序**（与 excavator_pose_reward 对齐）：
- **铲斗约定**：bucket 正 = 内收/开口平行地面；bucket 负 = 开口朝下。挖掘侧统一 swing（`swing_dig`），无段间多余旋转；卸料侧仅转到 `_DUMP_SWING=-1.15`（满足 abs≥1.0 即可）。
- **挖掘段**：低位下铲（_LOAD_NEAR 调低 boom/stick）→ 铲斗开口朝下 bucket=-0.85（bucket_out）→ 同高度 bucket<-1.0（bucket_in_scoop）→ **先收斗**（同低位 bucket=0.4 开口平行）→ **再抬臂**（secure→lift→carry）。
- **卸料段**：transport 只转到 -1.15；dump 段 bucket 经 0.5→0.08 以触发 dump_open 4.0。

**训练准入阈值（当前阶段）**：
- 使用 `--success_reward_threshold 4.0` 可要求轨迹达到 dump_open（abs_swing>=1.0 且 bucket>=THRESH_BUCKET_DUMPED）；使用 `3.0` 则仅要求回转到卸料侧并做卸料动作。
- 采集成功判定以 **replay 侧** 为准；`ee_replay` 对挖掘机已做保真跟踪（replay follower），固定 pose 下 replay 可达到 4.0。

**挖机采集路径（当前配置）**：
- `record_sim_episodes.py` 在 `equipment_model=excavator_simple` 时，支持两种流程：
  - `ee_replay`（默认）：先在 `EESimBackend` rollout，再在 `SimBackend` 用 **replay follower** 重放录制（每目标多步跟踪 + 卸料窗口 hold），保证 replay 与 EE 语义一致并可打满 4.0。
  - `direct_sim`：直接在 `SimBackend` rollout（用于快速验证与对照）。
- 默认使用 `ExcavatorTaskSpaceDigDumpPolicy`；`ExcavatorJointSpaceDigDumpPolicy` 保留为对照。

### 4.5 record_sim_episodes.py — 数据采集

- 当前为兼容壳层，转调 `runners/collect_runner.py`。
- 核心流程拆到 `testbed/pipelines.py`（`ee_replay`/`direct_sim`），HDF5 写入集中到 `io/hdf5_writer.py`。

**按设备分支的数据采集流程**：
1. 挖掘机（`excavator_simple`）可选两阶段 `ee_replay`（默认）或单阶段 `direct_sim`。`ee_replay` 时成功判定与 HDF5 内容均以 **replay 侧** 为准（replay follower 保证保真跟踪）。
2. 其他设备保持两阶段流程：`EESimBackend` rollout 生成关节轨迹，再在 `SimBackend` 重放录制 HDF5。

**挖掘机注意事项**：
- 挖掘机 `state_dim=4`（四个主关节），无夹爪维度；`record_sim_episodes.py` 不会对挖掘机轨迹执行“夹爪 ctrl 回填”。

**当前挖机默认策略**：
- `sim_lifting_cube_scripted + excavator_simple` 使用 `ExcavatorTaskSpaceDigDumpPolicy`。

**关键 CLI 参数**：
- `--task_name`：任务名（必须在 `SIM_TASK_CONFIGS` 中定义）
- `--dataset_dir`：输出目录
- `--num_episodes`：采集 episode 数
- `--equipment_model`：设备型号
- `--excavator_pipeline`：挖机流程选择（`ee_replay` 或 `direct_sim`，默认 `ee_replay`）
- `--success_reward_threshold`：可选成功阈值（默认使用任务 `max_reward`）
- `--fixed_excavator_box_pose`：挖掘机采集时使用固定方块位姿（用于可复现 smoke test）
- `--only_save_success`：只保存满足阈值的 episode
- `--target_success_episodes`：保存到指定成功条数后自动停止

### 4.6 policy.py — 策略封装

- 当前为兼容壳层，学习策略实现迁至 `policies/learned/act.py`。

**ACTPolicy(nn.Module)**：
- 构造时调用 `build_ACT_model_and_optimizer` 创建 DETRVAE 模型 + AdamW 优化器
- `__call__(qpos, image, actions=None, is_pad=None)`：
  - 训练时（`actions is not None`）→ 返回 `loss_dict = {l1, kl, loss}`
  - 推理时 → 返回 `a_hat: (B, chunk_size, state_dim)`
- 内部对 image 做 ImageNet 归一化

**CNNMLPPolicy(nn.Module)**：备选基线，未用于挖掘机。

### 4.7 detr/models/detr_vae.py — DETRVAE 模型

**CVAE 架构**：

```
训练时：
  Encoder:
    input = [CLS_embed(1,hidden) | qpos_embed(1,hidden) | action_embed(seq,hidden)]
    output → CLS token → latent_proj → μ(32) + logσ²(32)
    reparametrize → z(32)

  Decoder:
    input = image_features(ResNet18 → input_proj) + proprio(qpos → proj) + latent(z → proj)
    query = learned query_embed (num_queries 个)
    output → action_head → a_hat (num_queries, state_dim)
             is_pad_head → is_pad_hat

推理时：
    z = 0 向量（从先验采样退化为确定性）
    其余同 Decoder
```

**关键超参数**（当前挖掘机配置）：

| 参数 | 值 |
|------|-----|
| hidden_dim | 512 |
| dim_feedforward | 3200 |
| enc_layers | 4 |
| dec_layers | 7 |
| nheads | 8 |
| num_queries (chunk_size) | 100 |
| latent_dim | 32 |
| backbone | resnet18 |
| lr | 1e-5 |
| kl_weight | 10 |
| state_dim (excavator) | 4 |

### 4.8 imitate_episodes.py — 训练 & 评测入口

- 当前为兼容壳层，训练与评测逻辑迁至 `runners/train_runner.py` 与 `runners/eval_runner.py`。

**训练流程 `train_bc`**：
1. 创建 `ACTPolicy` → cuda
2. 可选从 `--resume_ckpt` 恢复（模型+优化器+epoch+min_val_loss）
3. 每 epoch：先 val（全 val set forward，记录 loss） → 再 train（逐 batch forward+backward）
4. 每 100 epoch 保存 checkpoint + 训练曲线图
5. 每 epoch 保存 `policy_latest.ckpt`
6. 结束保存 `policy_best.ckpt` + `policy_last.ckpt`

**评测流程 `eval_bc`**：
1. 加载 checkpoint + `dataset_stats.pkl`（归一化参数）
2. 创建 `SimBackend`（当前 `MuJoCoSimBackend`），执行 `--num_rollouts` 个 rollout（默认 50，可指定为 1 做快速验证）
3. 每步：`pre_process(qpos)` → `policy(qpos, image)` → `post_process(action)` → `backend.step(action)`
4. 支持 `--temporal_agg`：指数衰减加权融合历史 action chunk
5. 输出 success_rate / avg_return / 分级 reward 统计 / 视频

### 4.11 sim_backend.py — 仿真后端抽象层

- 当前为兼容壳层，统一后端迁至 `simulation/backends/mujoco.py`，接口为 `SimulationBackend`。

- `SimBackend`：统一 `reset/step/render/max_reward/set_initial_object_pose` 接口。
- `MuJoCoSimBackend`：对 `make_sim_env` 和 `BOX_POSE` 进行封装，向训练与采集流程提供稳定接口。
- 作用：将 ACT 训练/评测流程与 MuJoCo 实现细节解耦，后续接入新后端时只需新增 backend 实现。

### 4.12 ee_backend.py — EE 仿真后端抽象层

- 当前为兼容壳层，统一后端迁至 `simulation/backends/mujoco.py`。

- `EESimBackend`：统一 EE 录制阶段的 `reset/step/max_reward` 接口。
- `MuJoCoEESimBackend`：封装 `make_ee_sim_env`，让录制流程不直接依赖具体仿真实现。
- 作用：将演示采集入口（scripted rollout）与 EE 仿真实现细节解耦。

### 4.9 utils.py — 数据与工具

- **EpisodicDataset**：读 HDF5，随机采样时间步，归一化，输出 `(image, qpos, action, is_pad)`
- **load_data**：80/20 train/val 分割，构建 DataLoader
- **get_norm_stats**：计算 qpos/action 的 mean/std（跨所有 episode）
- **sample_box_pose / sample_box_pose_for_excavator / sample_insertion_pose**：随机物体位姿采样
- 挖掘机方块采样范围：x ∈ [3.4, 4.4]，y ∈ [-1, 1]，z = 0.25

### 4.10 visualize_episodes.py — 可视化

- `save_videos`：将 image_dict 或 image_list 拼接为 mp4（多相机横向拼接）
- `visualize_joints`：绘制 qpos vs action 对比曲线

---

## 5. 设备型号与 state_dim 映射

| equipment_model | state_dim | action 格式 | 关节 |
|-----------------|-----------|-------------|------|
| `vx300s_bimanual` | 14 | 左臂6+夹爪1+右臂6+夹爪1 | 双臂 ViperX |
| `vx300s_single` | 7 | 臂6+夹爪1 | 单臂 ViperX |
| `fairino5_single` | 7 | 臂6+夹爪1 | 法奥 FR5 |
| `excavator_simple` | 4 | swing+boom+stick+bucket | 挖掘机（无夹爪） |

> 统一入口：`config/registry.py` 中 `StateDimRegistry`。

---

## 6. MuJoCo 耦合点清单

当前代码与 `dm_control/mujoco` 的耦合集中在以下位置，未来解耦时需要抽象到 Backend 层：

| 文件 | 耦合点 | 调用举例 |
|------|--------|----------|
| `sim_env.py` | 模型加载 | `mujoco.Physics.from_xml_path(xml_path)` |
| `sim_env.py` | 环境创建 | `control.Environment(physics, task, ...)` |
| `sim_env.py` | 状态读取 | `physics.data.qpos`, `physics.data.qvel`, `physics.data.ctrl` |
| `sim_env.py` | 命名访问 | `physics.named.data.qpos[:16]`, `physics.named.data.qpos[joint_name]` |
| `sim_env.py` | 站点查询 | `physics.named.data.site_xpos['bucket_tip']`, `physics.named.data.site_xpos['dump_target']` |
| `sim_env.py` | 模型查询 | `physics.model.name2id()`, `physics.model.jnt_qposadr[]` |
| `sim_env.py` | 渲染 | `physics.render(height, width, camera_id)` |
| `sim_env.py` | 接触检测 | `physics.data.ncon`, `physics.data.contact[i].geom1/geom2` |
| `sim_env.py` | 重置 | `physics.reset_context()`, `physics.forward()` |
| `ee_sim_env.py` | mocap 控制 | `physics.data.mocap_pos[0]`, `physics.data.mocap_quat[0]` |
| `ee_sim_env.py` | 站点查询 | `physics.named.data.site_xpos['tcp_center']`, `physics.named.data.site_xpos['bucket_tip']`, `physics.named.data.site_xpos['dump_target']` |
| `ee_sim_env.py` | 四元数转换 | `mujoco.mju_mat2Quat(...)` |
| `sim_backend.py` | MuJoCo 封装 | `make_sim_env(...)`, `env._physics.render(...)`, `BOX_POSE[...]` |

---

## 7. 常用 CLI 命令

### 数据采集

```bash
python record_sim_episodes.py \
    --task_name sim_lifting_cube_scripted \
    --dataset_dir ./data_sim_episodes/sim_lifting_cube_scripted \
    --num_episodes 50 \
    --equipment_model excavator_simple \
    --excavator_pipeline ee_replay \
    --success_reward_threshold 4.0
```

```bash
# 挖掘机 smoke test（固定方块位姿，replay follower 下可打满 4.0）
python record_sim_episodes.py \
    --task_name sim_lifting_cube_scripted \
    --dataset_dir ./tmp_smoke_excavator \
    --num_episodes 1 \
    --equipment_model excavator_simple \
    --excavator_pipeline ee_replay \
    --fixed_excavator_box_pose \
    --success_reward_threshold 4.0
```

### 训练

```bash
python imitate_episodes.py \
    --task_name sim_lifting_cube_scripted \
    --ckpt_dir ./ckpts/excavator_act_v1 \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000 --lr 1e-5 --seed 0 \
    --equipment_model excavator_simple \
    --num_workers 4 --prefetch_factor 2 \
    --persistent_workers 1 --pin_memory 1
```

### 续训

```bash
# 追加 --resume_ckpt 即可，start_epoch 从 ckpt 自动推断
python imitate_episodes.py \
    ... (同训练参数) ... \
    --resume_ckpt ./ckpts/excavator_act_v1/policy_latest.ckpt
```

### 评测

```bash
python imitate_episodes.py \
    ... (同训练参数) ... \
    --equipment_model excavator_simple \
    --eval
# 快速验证（只跑 1 轮）
python imitate_episodes.py ... --eval --num_rollouts 1
```

---

## 8. 已知技术债与缺口

| 编号 | 问题 | 严重度 | 说明 |
|------|------|--------|------|
| T1 | 无土体物理 | 致命 | MuJoCo 刚体接触无法表达挖掘/装载核心物理 |
| T2 | 仿真后端解耦仍不彻底 | 中 | 已新增 `SimBackend` 封装评测/录制入口，`simulation/envs` 仍桥接 legacy `sim_env.py/ee_sim_env.py`（过渡态）；后续可逐步将 Task 逻辑迁入 `simulation/envs` 并废弃 legacy 直连 |
| T3 | 学习阶段路由尚未接入 | 中 | 已有挖机脚本分阶段轨迹，但 ACT 侧仍是单策略推理，无显式 Phase Router |
| T4 | 无安全约束 | 中 | 无 Guard、无关节限位校验、无异常退让 |
| T5 | 无状态接口层 | 中 | 策略直接读取原始 qpos，缺少结构化 S_t |
| T6 | 评测体系简陋 | 中 | 仅有 success_rate，缺少平滑度、时长等工程指标 |
| T7 | 模型可插拔能力有限 | 中 | 评测侧已通过后端接口解耦，但策略侧仍以 ACT/CNNMLP 为主，缺少统一 policy registry |
| T8 | `utils.py` 中 `load_data` 签名不一致 | 低 | 文件中定义接受 6 参数，但 `imitate_episodes.py` 调用时传了更多 kwargs |
| T9 | `BOX_POSE` 全局变量 | 低 | 跨模块通过全局列表传递方块位姿，容易出错 |

---

## 9. Checkpoint 文件格式

训练中保存的 checkpoint 包含：

```python
{
    'model_state_dict': policy.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': int,
    'min_val_loss': float,
    'config': {
        'task_name': str,
        'seed': int,
        'policy_class': str,
    }
}
```

评测时用 `policy_best.ckpt`（仅含 `model_state_dict` 或完整 dict，兼容两种格式）。

推理还需 `dataset_stats.pkl`：`{'qpos_mean', 'qpos_std', 'action_mean', 'action_std', 'example_qpos'}`。

---

## 10. 奖励函数设计

### ExcavatorSimpleLiftingCubeTask（Pose-only）

挖机奖励已切换为 **pose-only**：仅依据关节姿态（swing/boom/stick/bucket），不依赖刚体接触。

| reward | 阶段 | 关节条件（阈值在 excavator_pose_reward.py 中集中配置） |
|--------|------|------------------------------------------------------|
| 0.0 | start | 默认 |
| 1.0 | bucket_out | 铲斗接近（bucket < -0.6, abs_swing < 1.0） |
| 2.0 | bucket_in_scoop | 装载成立（bucket < -1.0） |
| 2.8 | swing_to_target | 回转到卸料侧（abs_swing >= 1.0, bucket < -0.4） |
| 3.0 | dump_open | 卸料动作（abs_swing >= 1.0 且 bucket >= THRESH_BUCKET_DUMP，约 -0.55） |
| 4.0 | dump_open | 卸料完成（abs_swing >= 1.0 且 **bucket >= THRESH_BUCKET_DUMPED，约 -0.31**） |

**达到 reward=4.0 的条件**：轨迹中至少有一帧同时满足 `abs(qpos[0]) >= 1.0`（回转至卸料侧）且 `qpos[3] >= THRESH_BUCKET_DUMPED`（当前约 -0.31，见 `excavator_pose_reward.py`）。`ee_replay` 使用 replay follower 后，固定 pose 下 replay 可达到 4.0；采集时可用 `--success_reward_threshold 4.0`。

- `simulation/rewards/excavator_pose_reward.py`：`ExcavatorPoseReward` 提供 `evaluate(qpos_traj)`（离线）和 `step_reward(qpos)`（环境每步），返回 `PoseRewardResult(reward, phases, phase_flags, phase_switch_steps)`。
- `testbed/evaluators.py` 调用上述模块，在 `info["pose_eval"]` 中输出 `pose_reward/pose_phases/pose_phase_flags/pose_phase_switch_steps`，用于可解释流程日志与回归对照。

---

## 11. 依赖关系图（模块间 import）

```
record_sim_episodes.py → runners/collect_runner → testbed/pipelines → simulation/backends
scripted_policy.py → policies/scripted ───────────────────────────┘
imitate_episodes.py → runners/train_runner|eval_runner → policies/learned → detr/*
sim_backend.py / ee_backend.py → simulation/backends (mujoco)
simulation/envs → sim_env.py / ee_sim_env.py → constants.py
io/hdf5_writer.py / io/stats.py → utils.py (dataset)
```

---

## 12. 开发规范

### 12.1 文档同步规则

**每次对 `PACT/` 目录下代码的修改，必须同步更新本文档中受影响的章节。** 具体包括但不限于：

- 新增/删除/重命名文件 → 更新 §2 目录结构
- 修改函数签名或接口 → 更新对应模块详解（§4）
- 修改数据格式 → 更新 §3 数据流
- 修改常量或配置 → 更新 §4.1
- 新增/修改 CLI 参数 → 更新 §7
- 修复技术债 → 更新 §8

### 12.2 代码风格

- 使用 Python 3.10+
- 类型注解鼓励但不强制
- 常量集中在 `constants.py`
- 新增设备型号需同时更新 `constants.py` + `sim_env.py` + `ee_sim_env.py` + `detr/models/detr_vae.py` 中的 state_dim 分支

### 12.3 数据约定

- HDF5 字段名保持 `/observations/qpos`、`/observations/qvel`、`/observations/images/{cam}`、`/action` 不变
- `state_dim` 由 `equipment_model` 唯一确定
- 归一化参数存储在 `dataset_stats.pkl`，与 checkpoint 放在同一 `ckpt_dir`
