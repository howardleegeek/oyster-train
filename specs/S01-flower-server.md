---
task_id: S01-flower-server
project: oyster-train
priority: 1
estimated_minutes: 45
depends_on: []
modifies: ["server/"]
executor: glm
---

## Goal
Build a Flower federated learning server with custom DiLoCo strategy for orchestrating 10,000 phone clients training Qwen2.5-1.5B with LoRA.

## Context
- Oyster has 10K Android phones (Unisoc T616, 6GB RAM, Mali-G57 GPU)
- Each phone trains locally with LoRA rank=4 on Qwen2.5-1.5B-Instruct (INT4)
- Phones sync every 500 local steps (DiLoCo protocol)
- Each sync uploads compressed LoRA deltas (~2-10KB)
- Server aggregates using outer optimizer (Nesterov momentum, lr=0.7)

## Deliverables

### server/flower_server.py
- Flower server using `flwr.server.start_server()`
- Custom DiLoCo strategy class inheriting `flwr.server.strategy.FedAvg`
- Config: min_fit_clients=10, min_available_clients=50, fraction_fit=0.1
- Server-side model: Qwen2.5-1.5B-Instruct skeleton (just parameter shapes, no weights needed)
- gRPC server on port 8080

### server/diloco_strategy.py
- Custom `DiLoCoStrategy(FedAvg)`:
  - `configure_fit()`: Tell clients to run 500 local steps before reporting
  - `aggregate_fit()`: Receive compressed LoRA deltas, decompress, apply outer optimizer
  - Outer optimizer: Nesterov momentum (β=0.9, lr=0.7) on aggregated pseudo-gradients
  - `configure_evaluate()`: Evaluate on held-out validation set
  - Track global round number, handle stragglers (timeout 300s)

### server/requirements.txt
- flwr[simulation]>=1.13
- torch>=2.0
- transformers>=4.40
- peft>=0.10
- numpy

### server/config.py
- Pydantic settings:
  - `flower_port: int = 8080`
  - `min_clients: int = 10`
  - `fraction_fit: float = 0.1`
  - `local_steps: int = 500`
  - `outer_lr: float = 0.7`
  - `outer_momentum: float = 0.9`
  - `round_timeout: int = 300`
  - `total_rounds: int = 100`
  - `model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"`
  - `lora_rank: int = 4`
  - `lora_alpha: int = 8`

## Constraints
- Python 3.10+
- All settings via environment variables or config.py
- Must handle client disconnection gracefully
- Log each round: participants, aggregation time, loss if available
- No UI code
- Write pytest tests in tests/test_server.py

## Acceptance Criteria
- [ ] `pip install -r server/requirements.txt` succeeds
- [ ] `python server/flower_server.py` starts listening on port 8080
- [ ] DiLoCoStrategy correctly aggregates mock LoRA deltas in test
- [ ] pytest tests/test_server.py passes
