---
task_id: S15-lewm-integration
project: oyster-train
priority: 1
estimated_minutes: 60
depends_on: []
modifies:
  - models/lewm_config.py
  - models/lewm_loader.py
  - simulation/lewm_client.py
  - server/lewm_server.py
  - models/__init__.py
executor: glm
---

# S15: LeWorldModel Integration into Oyster Federated Training

## 目标
将 LeWorldModel (JEPA world model, ~15M params) 集成到 oyster-train 联邦学习框架，
使手机能通过摄像头+IMU数据训练隐私保护的物理世界模型。

## 背景
- **LeWM**: Yann LeCun 组的 JEPA world model, 从像素学物理预测, 只需 ~15M params
- **oyster-train**: 为 Qwen2.5-1.5B 设计的联邦训练框架 (Flower + DiLoCo)
- **关键优势**: LeWM 比 Qwen 小 100x, 手机跑训练毫无压力 (~220MB vs ~4GB)

## 架构对比

| 维度 | Qwen2.5 (现有) | LeWM (新增) |
|------|----------------|-------------|
| 参数量 | 1.5B | ~15M |
| 训练内存 | ~4GB (INT4+LoRA) | ~220MB (FP16 full) |
| 通信 | LoRA delta (~2MB) | Full delta (~60MB → 压缩后 <200KB) |
| 输入 | Text tokens | Pixel frames + IMU |
| 用途 | 语言理解 | 物理世界预测 |

## 已完成的文件

### 1. `models/lewm_config.py` — Pydantic 配置
- `LeWMConfig`: encoder/predictor/sigreg/training/federation/data 全配置
- `get_ubs1_config()`: UBS1 手机优化 (MobileNetV3, depth=4, batch=2)
- `get_simulation_config()`: CPU 测试 (96px, depth=2)
- `get_gpu_config()`: GPU 全尺寸 (ViT-Tiny, 匹配论文)
- 内存估算: `estimated_memory_mb` 属性

### 2. `models/lewm_loader.py` — 模型构建
- **SIGReg**: 修复 CUDA 硬编码 → `device=proj.device`
- **LeWM class**: 完整 JEPA 模型 (encoder → projector → predictor → pred_proj)
- **Encoder 后端**: MobileNetV3-Small (手机) / ViT-Tiny (GPU)
- **API**: `load_lewm_model()`, `get_model_state()`, `set_model_state()`, `extract_delta()`
- **无 LoRA**: 15M 参数足够小, 全参数联邦训练

### 3. `simulation/lewm_client.py` — Flower 客户端
- `LeWMPhoneClient(fl.client.NumPyClient)`: 手机训练客户端
- `SyntheticWorldDataset`: 模拟数据 (生产环境替换为摄像头)
- `fit()`: local training → delta computation → compression
- 复用 `CompressionPipeline` (Top-K + SignSGD)

### 4. `server/lewm_server.py` — Flower 服务端
- 复用 `DiLoCoStrategy` (Nesterov momentum outer optimizer)
- `create_lewm_initial_parameters()`: 随机初始化 (<1s)
- 支持 simulation/production 两种模式

## 下一步 (待 dispatch)

### Phase 1: 验证 (本地)
- [ ] `python3 server/lewm_server.py simulation` 启动服务端
- [ ] 5 个 `LeWMPhoneClient` 模拟训练
- [ ] 确认 loss 下降 + 压缩比 >100x

### Phase 2: Android 集成
- [ ] QVAC 添加 PyTorch Mobile 导出 (TorchScript/ONNX)
- [ ] Android camera pipeline: CameraX → 224×224 帧 → tensor
- [ ] IMU sensor pipeline: SensorManager → [accel, gyro] → action tensor
- [ ] gRPC client 连接 Flower server

### Phase 3: 数据场景
- [ ] 室内导航 (预测下一帧 → 规划路径)
- [ ] 物体交互 (手触物体 → 预测物理结果)
- [ ] 场景理解 (学习用户日常环境的物理规律)

## 不要做
- 不改现有 Qwen2.5 代码路径
- 不改 DiLoCoStrategy 核心逻辑
- 不改 CompressionPipeline
- 不改任何基础设施代码
