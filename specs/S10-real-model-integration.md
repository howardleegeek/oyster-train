---
task_id: S10-real-model-integration
project: oyster-train
priority: 2
estimated_minutes: 60
depends_on: [S02, S04]
modifies: ["models/qwen25_loader.py", "models/quantization.py"]
executor: glm
---
## Goal
Integrate Qwen2.5-1.5B-Instruct with LoRA adapters, validate the full compression pipeline works with real model deltas (not mock data).

## Constraints
- Model: `Qwen/Qwen2.5-1.5B-Instruct` from HuggingFace
- LoRA config: rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
- Use peft library for LoRA injection
- Quantization: INT4 via bitsandbytes (bnb_4bit_compute_dtype=float16)
- Test compression roundtrip: extract LoRA deltas -> compress -> decompress -> apply back -> verify model outputs match within tolerance

## Deliverables
- `models/qwen25_loader.py` - Load Qwen2.5-1.5B with INT4 quantization + LoRA
- `models/quantization.py` - INT4 quantization utilities
- `tests/test_real_model.py` - Integration test (downloads model, ~3GB):
  - Load model + LoRA
  - Forward pass with sample input
  - Extract LoRA deltas
  - Compress (expect 28 layers * 4 projections * rank=4 * hidden=1536)
  - Decompress and apply back
  - Verify output difference < 1e-3
  - Log compression ratio and sizes

## Key interfaces
```python
# compressor/lora_extractor.py
deltas = extract_lora_delta(model, base_state_dict)  # -> List[np.ndarray]
apply_lora_delta(model, deltas, layer_info)

# compressor/pipeline.py
pipeline = CompressionPipeline(k_ratio=0.01)
compressed = pipeline.compress(deltas)
recovered = pipeline.decompress(compressed)
```

## Do NOT
- Modify compressor/ or server/ modules
- Fine-tune the model (just verify the pipeline)
- Use more than 8GB GPU memory
