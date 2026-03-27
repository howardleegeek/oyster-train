cmake_minimum_required(VERSION 3.20)
project(QVAC VERSION 1.0.0 LANGUAGES C CXX)

# QVAC (llama.cpp fork) - Custom CMake Overlay
# 
# This CMakeLists.txt provides additional build targets and
# configuration for the QVAC project, which is a customized
# version of llama.cpp for Android ARM64 with LoRA training support.
#
# Primary build is done via build_android.sh, which clones
# and builds the upstream llama.cpp repository.

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Source directories
set(LLAMACPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

# Check if llama.cpp exists
if(NOT EXISTS "${LLAMACPP_DIR}")
    message(STATUS "llama.cpp not found. Run ./build_android.sh to clone and build.")
    return()
endif()

# Find llama library
find_path(LLAMA_INCLUDE_DIR
    NAMES llama.h
    PATHS
        "${LLAMACPP_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/include"
)

find_library(LLAMA_LIBRARY
    NAMES llama
    PATHS
        "${LLAMACPP_DIR}/build-android"
        "${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/lib"
)

# LoRA training support
set(QVAC_ENABLE_LORA ON CACHE BOOL "Enable LoRA training support")

# GGUF format support (always enabled in llama.cpp)
set(QVAC_ENABLE_GGUF ON CACHE BOOL "Enable GGUF model format support")

# Vulkan GPU support
set(QVAC_ENABLE_VULKAN ON CACHE BOOL "Enable Vulkan GPU backend")

# QVAC library - wrapper around llama.cpp with customizations
add_library(qvac SHARED
    src/qvac.cpp
)

target_include_directories(qvac PRIVATE
    ${LLAMA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(qvac PRIVATE
    ${LLAMA_LIBRARY}
    # Vulkan libraries if enabled
    $<$<BOOL:${QVAC_ENABLE_VULKAN}>:vulkan>
)

target_compile_definitions(qvac PRIVATE
    QVAC_ENABLE_LORA=$<BOOL:${QVAC_ENABLE_LORA}>
    QVAC_ENABLE_GGUF=$<BOOL:${QVAC_ENABLE_GGUF}>
    QVAC_ENABLE_VULKAN=$<BOOL:${QVAC_ENABLE_VULKAN}>
    LLAMA_USE_VULKAN=$<BOOL:${QVAC_ENABLE_VULKAN}>
)

# QVAC command-line tools
if(QVAC_ENABLE_LORA)
    add_executable(qvac-finetune
        src/tools/qvac_finetune.cpp
    )
    
    target_include_directories(qvac-finetune PRIVATE
        ${LLAMA_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(qvac-finetune PRIVATE qvac)
    
    # Add install target for finetune
    install(TARGETS qvac-finetune
        RUNTIME DESTINATION bin
    )
endif()

# QVAC inference tool
add_executable(qvac-inference
    src/tools/qvac_inference.cpp
)

target_include_directories(qvac-inference PRIVATE
    ${LLAMA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(qvac-inference PRIVATE qvac)

# Install targets
install(TARGETS qvac
    LIBRARY DESTINATION lib
)

install(FILES
    ${LLAMA_INCLUDE_DIR}/llama.h
    DESTINATION include
)

# Install llama.cpp shared library if built
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/lib/libllama.so")
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/lib/libllama.so
        DESTINATION lib
    )
endif()

# Install finetune binary if built
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/bin/finetune")
    install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/bin/finetune
        DESTINATION bin
    )
endif()

# Install main inference binary if built
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/bin/main")
    install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/out/arm64-v8a/bin/main
        DESTINATION bin
        RENAME qvac-cli
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "QVAC Configuration Summary")
message(STATUS "==========================================")
message(STATUS "  Version: ${QVAC_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  LoRA Support: ${QVAC_ENABLE_LORA}")
message(STATUS "  GGUF Support: ${QVAC_ENABLE_GGUF}")
message(STATUS "  Vulkan Support: ${QVAC_ENABLE_VULKAN}")
message(STATUS "  llama.cpp: ${LLAMACPP_DIR}")
message(STATUS "==========================================")
message(STATUS "")
