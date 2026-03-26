---
task_id: S09-e2e-fl-simulation
project: oyster-train
priority: 1
estimated_minutes: 45
depends_on: [S01, S02, S03]
modifies: ["simulation/sim_orchestrator.py", "simulation/run_e2e.py"]
executor: glm
---
## Goal
Create a working end-to-end FL simulation: Flower server with DiLoCo strategy + 10 simulated phone clients + compression pipeline. Must complete at least 3 FL rounds successfully.

## Constraints
- Use existing modules: `server/`, `compressor/`, `simulation/`
- Flower 1.x gRPC protocol
- Tiny mock model (NOT real Qwen - that's S10's job): `create_tiny_lora_model(hidden_dim=256, lora_rank=4)`
- Each client does 5 local training steps per round (simulated with random gradients)
- Compression pipeline: LoRA delta -> TopK 1% -> SignSGD -> msgpack
- Server aggregates with DiLoCo outer optimizer (Nesterov momentum)
- Must handle client dropout (randomly skip 2/10 clients per round)

## Deliverables
- `simulation/run_e2e.py` - Main script that starts Flower server + 10 sim clients in separate threads
- `simulation/fl_client.py` - Flower NumPyClient implementation wrapping PhoneClient logic
- `tests/test_e2e_simulation.py` - Test that runs 3 rounds and validates:
  - All rounds complete without error
  - Compression ratio > 100x
  - Aggregated model weights change each round
  - Metrics logged per round (loss, num_clients, compression_ratio)

## Key interfaces (from existing code)
```python
# compressor/pipeline.py
pipeline = CompressionPipeline(k_ratio=0.01)
compressed_bytes = pipeline.compress(param_deltas)  # List[np.ndarray] -> bytes
recovered = pipeline.decompress(compressed_bytes)    # bytes -> List[np.ndarray]

# server/diloco_strategy.py
strategy = DiLoCoStrategy(min_fit_clients=5, min_available_clients=8)

# compressor/lora_extractor.py
from compressor.lora_extractor import create_tiny_lora_model
model = create_tiny_lora_model(hidden_dim=256, lora_rank=4)
```

## Do NOT
- Download or load real LLM models
- Modify existing source files in server/ or compressor/
- Add new pip dependencies beyond flower, torch, numpy
