---
task_id: S02-compression
project: oyster-train
priority: 1
estimated_minutes: 40
depends_on: []
modifies: ["compression/"]
executor: glm
---

## Goal
Build 3-layer gradient compression pipeline: LoRA delta extraction → Top-K sparsification → 1-bit SignSGD quantization. Target: compress each phone's update from ~3MB to 2-10KB.

## Context
- Each phone trains Qwen2.5-1.5B-Instruct with LoRA rank=4
- LoRA affects q_proj, k_proj, v_proj, o_proj in each transformer layer
- Qwen2.5-1.5B has 28 layers → 28 × 4 = 112 LoRA adapter pairs (A, B matrices)
- Raw LoRA delta: ~3MB (float32) or ~750KB (float16)
- Target after 3-layer compression: 2-10KB per sync

## Deliverables

### compression/lora_extractor.py
- `extract_lora_delta(model_before, model_after) -> Dict[str, Tensor]`
  - Compute diff of LoRA A and B matrices between checkpoints
  - Return dict mapping layer names to delta tensors
- `apply_lora_delta(model, deltas: Dict[str, Tensor])`
  - Apply aggregated deltas back to model

### compression/topk_sparse.py
- `topk_compress(deltas: Dict[str, Tensor], k_ratio: float = 0.01) -> CompressedDeltas`
  - Flatten all deltas, keep only top-K% by magnitude
  - Return: indices (int32) + values (float16) + shape metadata
  - Error feedback: accumulate residual for next round
- `topk_decompress(compressed: CompressedDeltas) -> Dict[str, Tensor]`
  - Reconstruct full-size tensors from sparse representation

### compression/signsgd.py
- `sign_compress(compressed: CompressedDeltas) -> SignCompressed`
  - Replace each value with its sign bit (1-bit) + global scale factor
  - Pack sign bits into uint8 arrays (8 values per byte)
  - Return: sign_bits (uint8 packed) + scale (float32) + indices + metadata
- `sign_decompress(signed: SignCompressed) -> CompressedDeltas`
  - Unpack sign bits, multiply by scale factor

### compression/pipeline.py
- `CompressionPipeline` class:
  - `compress(deltas: Dict[str, Tensor]) -> bytes`
    - Chain: LoRA delta → Top-K 1% → SignSGD → serialize to bytes
    - Track error feedback state internally
  - `decompress(data: bytes) -> Dict[str, Tensor]`
    - Reverse: deserialize → sign_decompress → topk_decompress
  - `get_compression_ratio() -> float`
    - Report actual compression ratio achieved
- Serialization: use msgpack or struct.pack for cross-platform safety

### compression/README.md
- Compression math explanation
- Expected sizes at each stage

## Constraints
- Python 3.10+
- Dependencies: torch, numpy, msgpack-python
- All operations must work on CPU (phones don't have CUDA)
- Deterministic: same input → same output
- Thread-safe (multiple clients compressing simultaneously on server)
- Use msgpack for serialization (safe for untrusted inputs from phone clients)

## Acceptance Criteria
- [ ] Pipeline compresses 3MB LoRA delta to <10KB
- [ ] Decompress(compress(x)) recovers top-1% values exactly (sign direction preserved)
- [ ] Compression ratio ≥ 300x reported correctly
- [ ] Error feedback accumulator works across rounds
- [ ] pytest tests/test_compression.py passes with random tensor inputs
- [ ] Benchmark: compress + decompress < 100ms on CPU
