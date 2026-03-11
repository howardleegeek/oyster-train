# Oyster Phone Training Protocol

Federated learning system for training Qwen2.5 on 10,000 Android phones with DiLoCo optimization and gradient compression.

## Overview

Oyster enables privacy-preserving model training on edge devices through:
- **DiLoCo (Distributed Low-Communication)**: Nesterov momentum optimizer reduces communication rounds
- **LoRA Adapters**: Efficient fine-tuning with rank-4 adapters
- **3-Layer Compression**: Top-K sparsification + 1-bit SignSGD achieves >300x compression
- **Phone-Optimized**: Designed for UBS1 phones (Unisoc T616, 6GB RAM, Mali-G57 GPU)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cloud Server                                │
│  ┌─────────────┐    ┌──────────────────────────────────────┐   │
│  │  Flower     │───▶│  DiLoCo Strategy                    │   │
│  │  Server     │    │  - Nesterov momentum (β=0.9, lr=0.7)│   │
│  │  (gRPC:8080)│    │  - Aggregate compressed LoRA deltas │   │
│  └─────────────┘    │  - Global model sync                 │   │
│                     └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC (2-10KB per update)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phone Clients (10,000)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Local Training (500 steps)                             │   │
│  │  ┌─────────┐   ┌──────────────┐   ┌──────────────────┐  │   │
│  │  │ Qwen2.5 │──▶│ LoRA(rank=4) │──▶│ Compress & Send │  │   │
│  │  │ (INT4)  │   │ ┌──┐ ┌──┐   │   │ ┌──────┐ ┌─────┐│  │   │
│  │  └─────────┘   │ │A │ │B │   │   │ │Top-K │ │Sign ││  │   │
│  │                │ └──┘ └──┘   │   │ │ 1%   │ │SGD  ││  │   │
│  │                │ 112 adapters  │   │ └──┬───┘ └──┬──┘│  │   │
│  │                └──────────────┘   │    │       │   │  │   │
│  │                                     ▼    ▼       ▼   │  │   │
│  │                                    3MB → 75KB → 10KB  │  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
make install
```

Or manually:
```bash
pip install flwr torch transformers peft numpy pyyaml msgpack-python pytest
```

### 2. Start Server

```bash
make server
```

Or:
```bash
python server/flower_server.py --port 8080 --rounds 100 --min-clients 10
```

### 3. Run Simulation

```bash
make sim
```

Or:
```bash
python simulation/sim_orchestrator.py --clients 10 --rounds 3
```

## Configuration

### Model Configuration

Use `models/qwen25_config.py` for hyperparameter management:

```python
from models.qwen25_config import get_ubs1_config, get_simulation_config

# Production config for UBS1 phones
config = get_ubs1_config()
# - Qwen2.5-1.5B-Instruct, INT4 quantization
# - LoRA rank=4, alpha=8
# - 500 local steps, 100 rounds
# - Top-K 1% + SignSGD compression

# Testing config (CPU-friendly)
test_config = get_simulation_config()
# - Qwen2.5-0.5B-Instruct
# - 10 local steps, 10 rounds
```

### Server Configuration

Configure via environment variables:

```bash
export OYSTER_FLOWER_PORT=8080
export OYSTER_MIN_CLIENTS=10
export OYSTER_OUTER_LR=0.7
export OYSTER_OUTER_MOMENTUM=0.9
```

Or use `ServerConfig`:

```python
from server.config import ServerConfig

config = ServerConfig(
    flower_port=9090,
    min_clients=5,
    outer_lr=0.5,
    outer_momentum=0.9
)
```

## Testing

Run all tests:

```bash
make test
```

Or specific test suites:

```bash
# Compression tests
pytest tests/test_compression.py -v

# Server tests
pytest tests/test_server.py -v

# Integration tests
pytest tests/test_integration.py -v
```

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **PoC** | Tiny model, 5 clients, CPU simulation | ✅ Complete |
| **Federated** | Real Qwen2.5-1.5B, 100 clients, GPU simulation | 🔨 In Progress |
| **Scale** | 1,000 clients, multi-server deployment | 📋 Planned |
| **Deploy** | 10,000 phones, production deployment | 📋 Planned |

## Project Structure

```
oyster-train/
├── client/              # Real phone client code (TBD)
├── compression/         # 3-layer gradient compression
│   ├── lora_extractor.py   # LoRA delta extraction
│   ├── topk_sparse.py      # Top-K sparsification
│   ├── signsgd.py          # 1-bit quantization
│   └── pipeline.py         # End-to-end pipeline
├── models/              # Model configuration
│   └── qwen25_config.py    # Pydantic config models
├── server/              # Flower server
│   ├── config.py           # Server settings
│   ├── diloco_strategy.py  # DiLoCo aggregation strategy
│   └── flower_server.py    # gRPC server entry point
├── simulation/          # Phone simulation harness
│   ├── data_loader.py      # Non-IID data shards
│   ├── sim_client.py       # Simulated phone client
│   └── sim_orchestrator.py # Simulation runner
├── tests/               # Test suites
│   ├── test_compression.py
│   ├── test_server.py
│   └── test_integration.py
├── specs/               # Task specifications
├── README.md
└── Makefile
```

## Key Components

### DiLoCo Strategy

The DiLoCo optimizer reduces communication by applying Nesterov momentum on the server side:

```
v_t = β * v_{t-1} - lr * (grad_t + β * v_{t-1})  # Lookahead gradient
θ_t = θ_{t-1} - v_t                              # Parameter update
```

Where:
- β = 0.9 (momentum coefficient)
- lr = 0.7 (outer learning rate)
- grad_t = weighted average of client pseudo-gradients

### Compression Pipeline

Each client's update (~3MB) is compressed to 2-10KB:

1. **LoRA Delta Extraction**: Extract only adapter weight changes
2. **Top-K Sparsification**: Keep top 1% by magnitude (100x compression)
3. **SignSGD Quantization**: 1-bit values with global scale (3x compression)

**Total**: >300x compression with error feedback for accuracy

### Non-IID Data Distribution

Simulation uses Dirichlet distribution (α=0.5) to create realistic data heterogeneity across clients, mimicking real-world phone user patterns.

## Performance Targets

- **Compression Ratio**: >300x (3MB → 10KB)
- **Communication Cost**: <100KB per phone per round
- **Training Latency**: <2 hours for 100 rounds (10K phones)
- **Memory per Phone**: <4GB with LoRA + INT4 quantization
- **Model Quality**: Target <5% accuracy drop vs centralized training

## License

Proprietary - Oyster Project
