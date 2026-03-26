---
task_id: S11-qvac-build-system
project: oyster-train
priority: 3
estimated_minutes: 45
depends_on: []
modifies: ["client/qvac/build.sh", "client/qvac/Makefile", "client/qvac/CMakeLists.txt"]
executor: glm
---
## Goal
Create a working QVAC (llama.cpp fork) cross-compilation pipeline for Android ARM64. Produce a build script that compiles llama.cpp with LoRA training support + Vulkan GPU backend for the UBS1 phone (Unisoc T616, Mali-G57).

## Constraints
- Base: llama.cpp latest stable (clone from https://github.com/ggerganov/llama.cpp)
- Target: Android API 28+, ARM64-v8a (Cortex-A75 + Cortex-A55)
- GPU: Vulkan 1.1 (Mali-G57 MP2) - NOT OpenCL
- NDK: r26b or later, CMake cross-compile
- Must include: LoRA fine-tuning capability (llama.cpp has `finetune` example)
- Output: `libllama.so` + `finetune` binary for Android ARM64
- GGUF model format support (for Qwen2.5-1.5B INT4)

## Deliverables
- `client/qvac/build_android.sh` - Complete build script:
  1. Clone llama.cpp if not present
  2. Set up NDK toolchain
  3. CMake configure with: -DLLAMA_VULKAN=ON -DLLAMA_BUILD_EXAMPLES=ON
  4. Cross-compile for arm64-v8a
  5. Output .so and binaries to `client/qvac/out/arm64-v8a/`
- `client/qvac/CMakeLists.txt` - Custom CMake overlay for LoRA training integration
- `client/qvac/README.md` - Build instructions and UBS1 hardware specs
- `tests/test_qvac_build.py` - Verify build script is syntactically correct, check required tools exist

## Do NOT
- Actually run the cross-compilation (no NDK on cluster nodes)
- Modify any Python modules
- Add GPU drivers or system packages
