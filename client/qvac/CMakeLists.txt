# Wrapper CMakeLists.txt for QVAC Fabric LLM (llama.cpp fork)
# Configures the build for Android ARM64 with Vulkan and NEON support

# Minimum CMake version
cmake_minimum_required(VERSION 3.20)

# Project name
project(QVAC_Fabric_LLM_Android)

# Set the source directory for llama.cpp
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

# Set optimization flags for Cortex-A75
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a75")

# Option to disable Vulkan (CPU-only fallback)
option(DISABLE_VULKAN "Disable Vulkan backend for CPU-only build" OFF)

# Add the llama.cpp subdirectory
add_subdirectory(${LLAMA_CPP_DIR})

# If Vulkan is disabled, turn off GGML_VULKAN
if(DISABLE_VULKAN)
    set_property(GLOBAL PROPERTY GGML_VULKAN OFF)
endif()

# Ensure LoRA training is enabled (if not already set by the subproject)
# Note: This assumes llama.cpp has LLAMA_LORA_TRAIN option. If not, we may need to patch.
# For now, we rely on the build script to set it via command line.