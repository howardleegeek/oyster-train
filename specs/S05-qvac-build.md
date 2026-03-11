---
task_id: S05-qvac-build
project: oyster-train
priority: 2
estimated_minutes: 45
depends_on: []
modifies: ["client/qvac/"]
executor: glm
---

## Goal
Set up QVAC Fabric LLM (llama.cpp fork with LoRA training) build system for cross-compiling to Android ARM (Unisoc T616, Mali-G57 GPU). Create build scripts and document the process.

## Context
- QVAC Fabric LLM is a llama.cpp fork by Tether Data that adds LoRA training with mobile GPU support
- Target: Android 14 on Unisoc T616 (ARM Cortex-A75 + A55, Mali-G57 MP1)
- Mali-G57 supports Vulkan 1.1 (not OpenCL well)
- llama.cpp already supports Android cross-compilation via CMake + NDK
- We need: inference + LoRA training on device

## Deliverables

### client/qvac/README.md
- Document QVAC Fabric LLM source location and build requirements
- Step-by-step cross-compilation guide for Android ARM64
- Mali-G57 Vulkan compatibility notes
- Fallback plan: CPU-only (NEON SIMD on Cortex-A75)

### client/qvac/build_android.sh
```bash
#!/bin/bash
# Cross-compile QVAC Fabric for Android ARM64
# Requires: Android NDK r26+, CMake 3.20+
# Target: Unisoc T616 (ARM Cortex-A75/A55, Mali-G57 Vulkan)
```
- Set NDK toolchain path
- CMake configure with:
  - `-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake`
  - `-DANDROID_ABI=arm64-v8a`
  - `-DANDROID_PLATFORM=android-34`
  - `-DGGML_VULKAN=ON` (for Mali-G57)
  - `-DGGML_NEON=ON` (ARM NEON SIMD)
  - `-DLLAMA_LORA_TRAIN=ON` (enable LoRA training)
- Build static library + CLI binary
- Output to client/qvac/build/

### client/qvac/CMakeLists.txt
- Wrapper CMakeLists that configures llama.cpp build
- Option to disable Vulkan (CPU-only fallback)
- Set optimization flags for Cortex-A75 (-mcpu=cortex-a75)

### client/qvac/test_build.sh
- Verify build output exists and is ARM64 binary
- Check linked libraries (Vulkan, NEON)
- Print binary size

### client/qvac/model_prep.py
- Python script to prepare Qwen2.5-1.5B-Instruct for phone deployment:
  1. Download from HuggingFace
  2. Convert to GGUF format (INT4 quantization)
  3. Configure LoRA adapter (rank=4, alpha=8)
  4. Estimate memory usage on 6GB device
  5. Output to client/qvac/models/

## Constraints
- This is a BUILD SYSTEM setup, not the actual cross-compilation (we don't have NDK on this node)
- Focus on scripts, documentation, and model preparation
- The actual build will happen on a machine with Android NDK
- Use llama.cpp conventions and build patterns
- Python 3.10+ for model_prep.py
- Dependencies for model_prep.py: transformers, huggingface_hub, gguf (pip package)

## Acceptance Criteria
- [ ] build_android.sh is syntactically correct and well-documented
- [ ] CMakeLists.txt configures correct flags for T616 target
- [ ] model_prep.py can download and quantize Qwen2.5-1.5B-Instruct to GGUF
- [ ] README documents full build process clearly
- [ ] Vulkan + CPU fallback paths are both documented
