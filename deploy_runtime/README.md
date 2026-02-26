# deploy_runtime

轻量级真机部署框架（当前支持 Fairino + 挖掘机接口骨架）。

## 目录

- `real_env.py`：统一环境接口（`reset/step/close`）
- `gateway.py`：安全过滤 + 固定控制周期（DT）
- `safety.py`：关节限幅与步进限幅
- `factory.py`：按 `equipment_model` 和 `backend` 构造适配器
- `sdk_clients.py`：Fairino SDK 客户端封装 + 挖掘机后端占位
- `camera_provider.py`：top 相机采集器（OpenCV）
- `adapters/*`：设备适配器层

## 快速验证

### 1) Mock 验证（不连真机）

```bash
python deploy_runtime/run_deploy.py --equipment_model fairino_fr5 --backend mock --steps 5
python deploy_runtime/run_deploy.py --equipment_model excavator_simple --backend mock --steps 5
```

### 2) Fairino SDK 健康检查（连真机）

```bash
python deploy_runtime/run_fairino_healthcheck.py --robot_ip 192.168.58.2
```

### 3) 用现有 eval 链路跑真机

```bash
python imitate_episodes.py \
  --eval \
  --ckpt_dir <ckpt_dir> \
  --policy_class ACT \
  --task_name sim_lifting_cube_scripted \
  --batch_size 8 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-5 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --kl_weight 10 \
  --equipment_model fairino_fr5 \
  --real_backend sdk \
  --fairino_robot_ip 192.168.58.2 \
  --camera_source 0
```

## 说明

- 当前 Fairino 状态按 7 维输出：前 6 维关节 + 第 7 维占位（默认 0）。
- `sdk` 后端需要安装 Fairino Python SDK 与 OpenCV。
- 挖掘机 `sdk` 后端目前是可插拔占位：需注入你自己的 `excavator_backend` 对象。