#!/bin/bash
# Cross-compile QVAC Fabric for Android ARM64
# Requires: Android NDK r26+, CMake 3.20+
# Target: Unisoc T616 (ARM Cortex-A75/A55, Mali-G57 Vulkan)

# Set NDK toolchain path - adjust this path as needed for your system
# Example: export NDK=/opt/android-ndk-r26b
if [ -z "$NDK" ]; then
  echo "Error: NDK environment variable not set"
  echo "Please set NDK to the path of your Android NDK installation"
  echo "Example: export NDK=/opt/android-ndk-r26b"
  exit 1
fi

# Check if llama.cpp source exists
if [ ! -d "llama.cpp" ]; then
  echo "Error: llama.cpp directory not found"
  echo "Please clone the QVAC Fabric LLM (llama.cpp fork) into client/qvac/llama.cpp"
  exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure CMake with Android cross-compilation
cmake ../llama.cpp \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DGGML_VULKAN=ON \
  -DGGML_NEON=ON \
  -DLLAMA_LORA_TRAIN=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build static library and CLI binary
cmake --build . --config Release -- -j$(nproc)

# Output location confirmation
echo "Build completed. Output files are in:"
pwd
ls -la llama.cpp libggml.so* libllama.so* 2>/dev/null || echo "No binary files found yet"

# Return to original directory
cd ..