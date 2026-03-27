---
task_id: S04-integration
project: oyster-train
priority: 1
estimated_minutes: 40
depends_on: ["S01", "S02", "S03"]
modifies: ["tests/", "models/", "README.md"]
executor: glm
---

## Goal
Integration tests + model configuration + project README. Wire together server, compression, and simulation into a working end-to-end pipeline.

## Deliverables

### models/qwen25_config.py
- Pydantic model for all training hyperparameters:
  - Model: name, quantization (INT4/INT8/FP16), max_seq_len
  - LoRA: rank, alpha, target_modules, dropout
  - Training: lr, batch_size, local_steps, warmup_steps
  - Federation: num_rounds, fraction_fit, min_clients, outer_lr, outer_momentum
  - Compression: topk_ratio, use_signsgd, error_feedback
- Factory method: `get_ubs1_config()` returns optimal config for UBS1 phone (6GB RAM)
- Factory method: `get_simulation_config()` returns CPU-friendly config for testing

### tests/test_integration.py
- End-to-end test:
  1. Start Flower server in background thread
  2. Create 5 simulated phone clients
  3. Each client does 10 local steps (not 500, for speed)
  4. Clients send compressed updates to server
  5. Server aggregates with DiLoCo
  6. Verify: global model parameters changed
  7. Verify: compression pipeline round-trips correctly
  8. Verify: loss decreased after 3 rounds
- Test with smallest possible model (Qwen2.5-0.5B or even a tiny random model)
- Must complete in < 5 minutes on CPU

### tests/test_compression.py
- Unit tests for each compression stage:
  - test_lora_delta_extraction: extract → apply → verify
  - test_topk_compress: verify only top-K% values retained
  - test_signsgd: verify sign bits packed/unpacked correctly
  - test_pipeline_roundtrip: compress → decompress → verify fidelity
  - test_error_feedback: residuals accumulate across rounds
  - test_compression_ratio: verify ≥ 300x

### tests/test_server.py
- Server unit tests:
  - test_diloco_strategy_init: strategy creates with correct config
  - test_aggregate_fit: mock client results → verify aggregation
  - test_outer_optimizer: Nesterov momentum update correct
  - test_straggler_timeout: slow client handled gracefully
  - test_min_clients: server waits for minimum clients

### README.md
- Project overview: Oyster Phone Training Protocol
- Architecture diagram (ASCII)
- Quick start guide:
  1. Install deps
  2. Start server
  3. Run simulation
- Configuration reference
- Phase roadmap (PoC → Federated → Scale → Deploy)

### Makefile
- `make install`: pip install all requirements
- `make test`: pytest all tests
- `make sim`: run simulation with 10 clients, 3 rounds
- `make server`: start Flower server
- `make lint`: black + ruff check

## Constraints
- Tests must be fast (< 5 min total on CPU)
- Use tiny model or random weights for integration test, not full Qwen2.5-1.5B
- All tests must pass independently (no test ordering dependency)
- README should be clear enough for a new engineer to run the project

## Acceptance Criteria
- [ ] `make install` succeeds
- [ ] `make test` passes all tests
- [ ] `make sim` runs 3 rounds with 10 clients and completes
- [ ] README is comprehensive and accurate
- [ ] Integration test proves end-to-end flow works
