<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flower-1.x-3366CC?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IiMzMzY2Q0MiLz48L3N2Zz4=" alt="Flower">
  <img src="https://img.shields.io/badge/Android-ARM64-3DDC84?logo=android&logoColor=white" alt="Android">
  <img src="https://img.shields.io/badge/License-Proprietary-red" alt="License">
</p>

# Oyster Phone Training Protocol

> **Federated learning on 10,000 phones** — training AI models on-device with privacy-preserving distributed optimization.

Oyster enables two federated training paradigms on Android phones:

| Model | Params | Use Case | Memory | Communication |
|-------|--------|----------|--------|---------------|
| **Qwen2.5-1.5B** | 1.5B (LoRA) | Language understanding | ~4GB (INT4) | ~10KB/round |
| **LeWorldModel** | ~15M (full) | Physical world prediction | ~220MB (FP16) | <200KB/round |

---

## Core Technology

```
                         Cloud Server (DiLoCo)
                    ┌───────────────────────────┐
                    │   Flower gRPC :8080        │
                    │   Nesterov Momentum        │
                    │   β=0.9, lr=0.7            │
                    │   Aggregate → Broadcast    │
                    └─────────┬─────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Phone 1         Phone 2    ...  Phone 10,000
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Local Train  │  │ Local Train  │  │ Local Train  │
    │ 200-500 steps│  │ 200-500 steps│  │ 200-500 steps│
    │      ↓       │  │      ↓       │  │      ↓       │
    │ Δ → TopK(1%) │  │ Δ → TopK(1%) │  │ Δ → TopK(1%) │
    │ → SignSGD    │  │ → SignSGD    │  │ → SignSGD    │
    │ = >300x comp │  │ = >300x comp │  │ = >300x comp │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Key Innovations

- **DiLoCo**: Distributed Low-Communication optimizer — clients train locally for hundreds of steps before syncing, reducing communication rounds by 100x
- **3-Layer Compression**: LoRA delta extraction → Top-K sparsification (1%) → 1-bit SignSGD = **>300x compression** with error feedback
- **LeWM Integration**: JEPA world model learns physical environment prediction from phone cameras — only 15M parameters, trainable in ~220MB RAM

---

## Quick Start

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install flwr torch torchvision transformers peft einops numpy pyyaml msgpack-python
```

### Run Simulation (Qwen2.5)

```bash
# Terminal 1: Start server
python server/flower_server.py

# Terminal 2: Run 10 simulated phone clients
python simulation/sim_orchestrator.py --clients 10 --rounds 3
```

### Run Simulation (LeWM World Model)

```bash
# Terminal 1: Start LeWM server
python server/lewm_server.py simulation

# Terminal 2: Quick local test
python -c "
from models.lewm_config import get_simulation_config
from models.lewm_loader import load_lewm_model
import torch

model = load_lewm_model(get_simulation_config())
pixels = torch.randn(4, 8, 3, 96, 96)
actions = torch.randn(4, 8, 6)
out = model(pixels, actions)
print(f'Loss: {out[\"loss\"].item():.4f}')
"
```

---

## Model Configurations

### Qwen2.5 (Language)

```python
from models import get_ubs1_config

config = get_ubs1_config()
# Qwen2.5-1.5B-Instruct, INT4 quantization
# LoRA rank=4, alpha=8, targets: q/k/v/o_proj
# 500 local steps, batch=4
```

### LeWorldModel (Vision/Physics)

```python
from models import get_lewm_ubs1_config, get_lewm_simulation_config

# For UBS1 phones (6GB RAM)
config = get_lewm_ubs1_config()
# MobileNetV3 encoder, 4-layer predictor
# ~8M params, ~120MB training memory
# Camera frames (224×224) + IMU as actions

# For CPU testing
config = get_lewm_simulation_config()
# 96×96 images, 2-layer predictor, ~4M params
```

---

## Project Structure

```
oyster-train/
├── models/                    # Model backends
│   ├── qwen25_config.py          # Qwen2.5 configuration
│   ├── qwen25_loader.py          # Qwen2.5 + LoRA loader
│   ├── lewm_config.py            # LeWM configuration (NEW)
│   ├── lewm_loader.py            # LeWM JEPA model (NEW)
│   └── quantization.py           # INT4/INT8/FP16 quantization
├── server/                    # Flower FL server
│   ├── diloco_strategy.py        # DiLoCo aggregation strategy
│   ├── flower_server.py          # Qwen2.5 server
│   └── lewm_server.py            # LeWM server (NEW)
├── simulation/                # Phone simulation
│   ├── sim_client.py             # Qwen2.5 phone client
│   ├── lewm_client.py            # LeWM phone client (NEW)
│   └── sim_orchestrator.py       # Simulation runner
├── compressor/                # Gradient compression
│   ├── pipeline.py               # End-to-end compression
│   ├── topk_sparse.py            # Top-K sparsification
│   ├── signsgd.py                # 1-bit quantization
│   └── lora_extractor.py         # LoRA delta extraction
├── client/                    # Real phone clients
│   ├── android/                  # Android app (Kotlin)
│   └── qvac/                     # QVAC native build (C++)
├── deploy/                    # Deployment configs
├── tests/                     # Test suites
├── specs/                     # Task specifications
└── data/                      # Training data configs
```

---

## Technical Details

### DiLoCo Outer Optimizer

```
v_t = β · v_{t-1} - lr · (Δ_t + β · v_{t-1})   # Nesterov lookahead
θ_t = θ_{t-1} - v_t                               # Global update
```

### LeWM Architecture (JEPA)

Based on [LeWorldModel](https://arxiv.org/abs/2603.19312) (Maes, Le Lidec, Scieur, LeCun, Balestriero 2026):

```
Camera Frames → MobileNetV3 Encoder → Projector → Embeddings (B, T, 192)
                                                        ↓
IMU Data → Action Encoder ──────────────→ AR Predictor (6-layer Transformer)
                                                        ↓
                                              Predicted Next Embedding
                                                        ↓
                                    Loss = MSE(pred, target) + λ · SIGReg
```

- **SIGReg**: Sketch Isotropic Gaussian Regularizer — prevents representation collapse using characteristic function matching
- **No EMA, no pretrained encoder, no auxiliary losses** — trains stably end-to-end from random init

### Hardware Target: UBS1 Phone

| Spec | Value |
|------|-------|
| SoC | Unisoc T616 |
| CPU | 2× A75 @ 1.8GHz + 6× A55 @ 1.6GHz |
| GPU | Mali-G57 MP2 (Vulkan 1.1) |
| RAM | 4GB / 6GB LPDDR4X |
| Storage | 64GB / 128GB eMMC |

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| PoC | Tiny model, 5 clients, CPU simulation | ✅ Complete |
| LeWM | World model integration, phone-feasible | ✅ Complete |
| Federated | Real Qwen2.5-1.5B, 100 GPU clients | 🔨 In Progress |
| Android | Camera + IMU pipeline, on-device LeWM | 📋 Planned |
| Scale | 1,000 clients, multi-server | 📋 Planned |
| Deploy | 10,000 phones, production | 📋 Planned |

---

## Performance Targets

| Metric | Qwen2.5 | LeWM |
|--------|---------|------|
| Compression | >300x | >300x |
| Per-round comm | <10KB | <200KB |
| Training memory | <4GB | <250MB |
| 100 rounds (10K phones) | <2h | <30min |
| Quality vs centralized | <5% drop | TBD |

---

## License

Proprietary — Oyster Labs
