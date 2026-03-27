# QVAC - Quantized Video-Assisted Copilot

QVAC is a customized build of [llama.cpp](https://github.com/ggerganov/llama.cpp) for Android ARM64 devices, specifically optimized for the UBS1 phone.

## UBS1 Phone Hardware Specifications

### CPU
- **SoC**: Unisoc T616
- **Architecture**: ARMv8-A
- **Big Core**: 2x Cortex-A75 @ 1.8 GHz
- **Little Core**: 6x Cortex-A55 @ 1.6 GHz
- **ISA**: ARM64-v8a

### GPU
- **Model**: Mali-G57 MP2
- **Architecture**: Valhall (Arm Immortalis series)
- **Vulkan Support**: Vulkan 1.1+
- **Compute Shaders**: Supported (via Vulkan)
- **Memory**: Shared with system RAM

### Memory
- **RAM**: 4GB / 6GB LPDDR4X
- **Storage**: 64GB / 128GB eMMC

### Android Version
- **Target API**: 28+ (Android 9.0+)

## Features

- **GGUF Model Format**: Support for quantized models (Qwen2.5-1.5B INT4)
- **LoRA Fine-Tuning**: On-device fine-tuning capability via the `finetune` binary
- **Vulkan GPU Acceleration**: Hardware acceleration via Mali-G57
- **Cross-Compiled**: Built for Android ARM64 from Linux/macOS/Windows hosts

## Prerequisites

### Build Machine Requirements
- **Operating System**: Linux, macOS, or Windows (WSL2)
- **CMake**: Version 3.20 or later
- **Ninja**: Build system
- **Git**: Version control
- **Android NDK**: r26b or later

### Installing Android NDK

#### Linux/macOS
```bash
# Download NDK r26b
wget https://dl.google.com/android/repository/android-ndk-r26b-linux.zip

# Extract
unzip android-ndk-r26b-linux.zip

# Set environment variable
export ANDROID_NDK_HOME=/path/to/android-ndk-r26b
```

#### Via Android Studio
1. Open Android Studio
2. Go to SDK Manager
3. Install NDK (Side by side) - version 26.2.11394342 or later

## Building

### Quick Start

```bash
cd client/qvac
./build_android.sh
```

This will:
1. Clone the latest llama.cpp repository
2. Configure CMake for Android ARM64
3. Build with Vulkan and LoRA support enabled
4. Output binaries to `out/arm64-v8a/`

### Custom NDK Path

```bash
./build_android.sh /path/to/android-ndk-r26b
```

### Custom API Level

```bash
./build_android.sh /path/to/android-ndk-r26b 29
```

### Build Configuration

The build script configures llama.cpp with the following options:

| Option | Value | Description |
|--------|-------|-------------|
| Target | arm64-v8a | ARM64 Android architecture |
| API Level | 28+ | Android 9.0+ |
| STL | c++_shared | Shared C++ runtime |
| Vulkan | ON | GPU acceleration for Mali-G57 |
| Examples | ON | Build finetune and other tools |
| Tests | OFF | Skip test builds |
| Server | OFF | Skip server builds |
| Build Type | Release | Optimized release build |

## Output Files

After a successful build, the following files are available in `out/arm64-v8a/`:

```
out/arm64-v8a/
├── bin/
│   ├── finetune           # LoRA fine-tuning tool
│   ├── main               # Inference CLI
│   ├── quantize           # Model quantization
│   └── export-lora        # LoRA adapter export
├── lib/
│   └── libllama.so        # Main shared library
└── include/
    ├── llama.h            # Main API header
    └── llama/
        └── ...            # Additional headers
```

## Installing on UBS1 Phone

### Using ADB

```bash
# Connect phone via USB with USB debugging enabled
adb devices

# Push files to phone
adb push out/arm64-v8a/lib/libllama.so /data/local/tmp/
adb push out/arm64-v8a/bin/* /data/local/tmp/

# Set executable permissions
adb shell chmod +x /data/local/tmp/finetune
adb shell chmod +x /data/local/tmp/main

# Test inference
adb shell /data/local/tmp/main -m model.gguf -p "Hello, how are you?"

# Test LoRA fine-tuning
adb shell /data/local/tmp/finetune --model model.gguf --lora-output my_adapter.gguf --train-data train.txt
```

## Supported Models

### Qwen2.5-1.5B INT4
- **File**: `qwen2.5-1.5b-instruct-q4_0.gguf`
- **Size**: ~1 GB
- **RAM Required**: ~2-3 GB (shared with Vulkan)

### Other GGUF Models
QVAC supports any GGUF model compatible with llama.cpp, including:
- Llama 2/3 variants
- Mistral
- Phi
- Gemma
- Other quantized models

## LoRA Fine-Tuning

QVAC includes the `finetune` example from llama.cpp, enabling on-device fine-tuning using LoRA (Low-Rank Adaptation).

### Basic LoRA Training

```bash
# Prepare training data (one example per line)
echo "User: Hello\nAssistant: Hi there!" > train.txt

# Fine-tune on device
adb shell /data/local/tmp/finetune \
  --model qwen2.5-1.5b-instruct-q4_0.gguf \
  --lora-output adapter.gguf \
  --train-data train.txt \
  --lora-r 16 \
  --lora-alpha 32 \
  --train-epochs 3

# Use adapter for inference
adb shell /data/local/tmp/main \
  --model qwen2.5-1.5b-instruct-q4_0.gguf \
  --lora-adapter adapter.gguf \
  --prompt "Hello, how are you?"
```

## Vulkan GPU Acceleration

QVAC automatically uses Vulkan for GPU acceleration when available on the UBS1 phone.

### Vulkan Verification

```bash
# Check Vulkan support on device
adb shell dumpsys package | grep vulkan

# Verify Vulkan layers (optional)
adb shell ls /data/local/tmp/vulkan/
```

### Troubleshooting GPU

If GPU acceleration isn't working:
1. Ensure Vulkan drivers are up to date
2. Check GPU memory availability with `adb shell cat /proc/meminfo`
3. Reduce model size or batch size to fit available memory

## Performance Tips

### For Mali-G57 MP2
- **Batch Size**: 1-2 tokens
- **Context Length**: 2048 tokens optimal
- **Quantization**: Q4_0 or Q4_K_M recommended
- **Memory**: Use INT4 models to fit in 4-6GB RAM

### For Better Performance
- Close other apps while running
- Use Performance Mode (if available in phone settings)
- Reduce context length for faster inference
- Use smaller batch sizes for real-time applications

## Troubleshooting

### Build Issues

**NDK not found:**
```bash
# Set ANDROID_NDK_HOME explicitly
export ANDROID_NDK_HOME=/path/to/android-ndk-r26b
./build_android.sh
```

**CMake toolchain error:**
```bash
# Verify NDK version is r26b or later
cat $ANDROID_NDK_HOME/source.properties
```

**Git clone fails:**
```bash
# Check network connectivity
curl -I https://github.com/ggerganov/llama.cpp

# Or clone manually
cd client/qvac
git clone https://github.com/ggerganov/llama.cpp.git
./build_android.sh
```

### Runtime Issues on Phone

**libllama.so not found:**
```bash
# Set LD_LIBRARY_PATH
adb shell export LD_LIBRARY_PATH=/data/local/tmp
```

**Permission denied:**
```bash
# Make binaries executable
adb shell chmod +x /data/local/tmp/*
```

**Vulkan not available:**
```bash
# Check if Vulkan is supported
adb shell dumpsys SurfaceFlinger | grep vulkan
```

## Project Structure

```
client/qvac/
├── build_android.sh          # Main build script
├── CMakeLists.txt             # Custom CMake overlay
├── README.md                  # This file
├── llama.cpp/                 # Cloned llama.cpp (generated)
│   ├── include/
│   ├── src/
│   └── examples/
├── out/                       # Build output
│   └── arm64-v8a/
│       ├── bin/
│       ├── lib/
│       └── include/
└── src/                       # Custom QVAC sources
    ├── qvac.cpp
    └── tools/
        ├── qvac_finetune.cpp
        └── qvac_inference.cpp
```

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Base implementation
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Model format
- [Mali-G57 Specs](https://developer.arm.com/Processors/Mali-G57) - GPU details
- [Unisoc T616](https://www.unisoc.com/) - SoC details

## License

QVAC inherits the license from llama.cpp (MIT License).
