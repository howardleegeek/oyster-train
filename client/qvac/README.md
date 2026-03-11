# QVAC Fabric LLM Build System for Android ARM64

## Source Location
QVAC Fabric LLM is a llama.cpp fork by Tether Data that adds LoRA training with mobile GPU support.
The source should be placed in `client/qvac/llama.cpp` (or adjust paths accordingly).

## Build Requirements
- Android NDK r26+
- CMake 3.20+
- Python 3.10+ (for model preparation)
- Required Python packages: transformers, huggingface_hub, gguf

## Cross-compilation Guide for Android ARM64 (Unisoc T616)

### Step 1: Install Dependencies
```bash
# Install Android NDK (r26+)
# Install CMake (3.20+)
# Install Python packages for model preparation:
pip install transformers huggingface_hub gguf
```

### Step 2: Prepare the Model
Run the model preparation script to download and quantize the model:
```bash
cd client/qvac
python model_prep.py
```
This will download Qwen2.5-1.5B-Instruct from HuggingFace, convert to GGUF (INT4), and configure a LoRA adapter.

### Step 3: Build for Android
```bash
cd client/qvac
chmod +x build_android.sh
./build_android.sh
```
The build will output to `client/qvac/build/`.

### Step 4: Verify the Build
```bash
chmod +x test_build.sh
./test_build.sh
```

## Target Hardware Specifications
- **CPU**: Unisoc T616 (ARM Cortex-A75 + Cortex-A55, ARMv8.2-A)
- **GPU**: Mali-G57 MP1 (Vulkan 1.1 supported)
- **Android Version**: 14 (API level 34)

## CMake Configuration Flags Used
- `-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake`
- `-DANDROID_ABI=arm64-v8a`
- `-DANDROID_PLATFORM=android-34`
- `-DGGML_VULKAN=ON` (for Mali-G57 GPU acceleration)
- `-DGGML_NEON=ON` (ARM NEON SIMD for Cortex-A75)
- `-DLLAMA_LORA_TRAIN=ON` (enable LoRA training)

## Optimization for Cortex-A75
The wrapper CMakeLists.txt sets `-mcpu=cortex-a75` for optimized code generation.

## Fallback Plan: CPU-only
If Vulkan is not available or desired, set `-DGGML_VULKAN=OFF` in the build script.
The system will fall back to CPU execution with NEON SIMD acceleration.

## Memory Usage Estimation
For Qwen2.5-1.5B-Instruct quantized to INT4:
- Model size: ~0.5 GB
- Runtime memory: ~1.5 GB (including KV cache and overhead)
- Suitable for devices with 3GB+ RAM (targeting 6GB devices)

## LoRA Training Configuration
- Rank: 4
- Alpha: 8
- Target modules: QKV projections (standard for Llama-like models)

## Troubleshooting
1. **Build fails**: Ensure NDK and CMake versions meet requirements
2. **Vulkan not working**: Check device drivers and Vulkan support
3. **Model loading errors**: Verify model files are correctly quantized and placed in `client/qvac/models/`
4. **Performance issues**: Ensure NEON is enabled and consider adjusting LoRA parameters

## License
QVAC Fabric LLM is licensed under the Apache 2.0 license. See the source repository for details.