# Deploy Demo（Fairino 实机部署最小集）

本目录用于把 `ACT` 策略接到 Fairino 实机，支持：
- 指定关节 MoveJ 调试
- 错误复位
- 实时策略推理回放（可先规避相机/夹爪）

## 文件说明
- `fairino_env.py`：实机环境适配（RPC连接、读关节、MoveJ/ServoJ、软限位、安全拦截）
- `run_real_policy.py`：在线推理回放主入口（ACT）
- `run_movej_pose.py`：MoveJ 到指定关节角（调姿、对齐起始位）
- `run_reset_errors.py`：清错并重新使能
- `run_simple_demo.py`：无策略正弦扰动烟测
- `camera_stub.py`：相机占位（黑图）
- `safety.py`：限位/大步长检测

## 推荐联调顺序
1. **清错**
   - 运行 `run_reset_errors.py`
2. **调姿**
   - 运行 `run_movej_pose.py` 把实机对齐到训练初始位
3. **短步数回放**
   - 运行 `run_real_policy.py`，先小步数、低速验证
4. **逐步放开参数**
   - 提高 `num_steps`，再提高速度

## 先规避相机/夹爪可以吗？
可以，但有边界：
- **机械臂链路验证可以**：连接、推理、下发、轨迹平滑都可验证。
- **任务成功率不可靠**：抓取/放置任务强依赖视觉与夹爪反馈。

当前代码中：
- 相机默认使用 `camera_stub`（黑图）
- 夹爪未接入时，`run_real_policy.py` 用 `--gripper_input_value` 给第7维占位

## run_real_policy 关键参数
- `--robot_ip`：控制器 IP（默认 `192.168.58.2`）
- `--ckpt_dir`：包含 `policy_best.ckpt` 和 `dataset_stats.pkl`
- `--use_servo`：使用 ServoJ
- `--max_servo_step_rad`：每步最大关节变化（小一些更平滑更安全）
- `--servo_period_s`：控制周期
- `--smoothing_alpha`：低通平滑系数
- `--query_every`：每隔多少步重算一次策略输出（ACT分块消费）
- `--move_to_init --init_qpos ...`：先回初始姿态
- `--gripper_input_value`：无夹爪时给策略的占位值

## 已知问题排查
- `MoveJ/ServoJ error: 14`：多见于控制器拒绝执行（限位/速度/状态）
  - 先查 `GetRobotErrorCode`
  - 清错并使能
  - 降低 `speed/acc` 与 `max_servo_step_rad`
- 机械臂“顿挫”
  - 关掉高频打印
  - 降低 `max_servo_step_rad`
  - 使用固定周期 + 分块查询（已在脚本中）
