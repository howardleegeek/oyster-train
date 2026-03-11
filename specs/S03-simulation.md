---
task_id: S03-simulation
project: oyster-train
priority: 1
estimated_minutes: 50
depends_on: []
modifies: ["simulation/"]
executor: glm
---

## Goal
Build a phone simulation harness that simulates N Flower clients each performing local LoRA fine-tuning on Qwen2.5-1.5B-Instruct. This validates the federated training pipeline without real phones.

## Context
- Real deployment: 10K phones with 6GB RAM, Unisoc T616
- Simulation: each "phone" is a Flower client process training with LoRA
- Simulate resource constraints: limit memory per client, slow training speed
- Use Flower's built-in simulation capabilities where possible
- Each simulated phone gets a shard of training data (non-IID distribution)

## Deliverables

### simulation/sim_client.py
- Flower client class `PhoneClient(flwr.client.NumPyClient)`:
  - `get_parameters()`: Return current LoRA parameters
  - `fit(parameters, config)`:
    1. Load Qwen2.5-1.5B-Instruct with LoRA (from HuggingFace peft)
    2. Apply received global LoRA params
    3. Train for `config["local_steps"]` steps on local data shard
    4. Extract LoRA delta (difference from received params)
    5. Compress delta using CompressionPipeline (import from compression/)
    6. Return compressed parameters + num_examples
  - `evaluate(parameters, config)`: Evaluate on local validation shard, return loss + accuracy
- Memory simulation: log peak memory usage per training step

### simulation/data_loader.py
- `create_non_iid_shards(dataset, num_clients, alpha=0.5) -> List[Dataset]`
  - Use Dirichlet distribution (alpha parameter) to create non-IID data splits
  - alpha=0.5 = moderately heterogeneous (realistic for phone users)
  - Dataset: use a small dataset for simulation (wikitext-2 or alpaca-cleaned)
- `PhoneDataset` class:
  - Wraps a shard with proper tokenization for Qwen2.5
  - Max sequence length: 256 tokens (phone memory constraint)

### simulation/sim_orchestrator.py
- `run_simulation(num_clients=100, num_rounds=10)`:
  - Use `flwr.simulation.start_simulation()` if available
  - OR manually spawn N client processes connecting to Flower server
  - Configure: non-IID data, client sampling fraction, local steps
  - Log per-round: global loss, accuracy, communication bytes, wall time
  - Save training curves to `simulation/results/`
- CLI interface: `python simulation/sim_orchestrator.py --clients 100 --rounds 10`

### simulation/configs/default.yaml
```yaml
num_clients: 100
num_rounds: 10
local_steps: 500
batch_size: 4
max_seq_len: 256
lora_rank: 4
lora_alpha: 8
model_name: "Qwen/Qwen2.5-1.5B-Instruct"
data_alpha: 0.5
dataset: "wikitext"
compression: true
topk_ratio: 0.01
```

## Constraints
- Python 3.10+, PyTorch required
- Must work on CPU (GCP nodes don't have GPU)
- Each simulated client should use < 4GB RAM (matching phone constraint)
- Use HuggingFace transformers + peft for model loading
- Non-IID distribution is critical (phones have different user data)
- If full Qwen2.5-1.5B is too heavy for CPU simulation, use Qwen2.5-0.5B as stand-in with same architecture
- Log compression ratio and bytes transmitted per round

## Acceptance Criteria
- [ ] `python simulation/sim_orchestrator.py --clients 10 --rounds 3` completes
- [ ] Non-IID data shards created correctly (verify distribution skew)
- [ ] Each client trains locally and sends compressed updates
- [ ] Global model improves over rounds (loss decreases)
- [ ] Training curves saved to simulation/results/
- [ ] Peak memory per client logged and < 4GB
