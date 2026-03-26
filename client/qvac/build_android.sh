#!/usr/bin/env bash
#
# QVAC (llama.cpp fork) Android ARM64 Build Script
#
# Cross-compiles llama.cpp with:
# - Vulkan GPU backend for Mali-G57 (UBS1 phone)
# - LoRA fine-tuning support (finetune example)
# - GGUF model format support (Qwen2.5-1.5B INT4)
#
# Target: Android API 28+, ARM64-v8a (Cortex-A75 + Cortex-A55)
# NDK: r26b or later
#
# Usage:
#   ./build_android.sh [/path/to/ndk] [api_level]
#
# Example:
#   ./build_android.sh
#   ./build_android.sh /opt/android-ndk-r26b 29

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
LLAMACPP_DIR="${SCRIPT_DIR}/llama.cpp"
BUILD_DIR="${LLAMACPP_DIR}/build-android"
OUTPUT_DIR="${SCRIPT_DIR}/out/arm64-v8a"

# Default NDK path - try common locations
DEFAULT_NDK_PATHS=(
    "${HOME}/Android/Sdk/ndk/26.2.11394342"
    "${HOME}/Android/Sdk/ndk-bundle"
    "/opt/android-ndk-r26b"
    "/usr/local/android-ndk"
)

# Android API level
DEFAULT_API_LEVEL="28"

# Parse arguments
NDK_PATH="${1:-}"
API_LEVEL="${2:-$DEFAULT_API_LEVEL}"

# Function to find NDK
find_ndk() {
    if [ -n "${NDK_PATH}" ] && [ -d "${NDK_PATH}" ]; then
        echo "${NDK_PATH}"
        return 0
    fi

    for path in "${DEFAULT_NDK_PATHS[@]}"; do
        if [ -d "${path}" ]; then
            echo "${path}"
            return 0
        fi
    done

    # Try to find via ANDROID_NDK_HOME or ANDROID_NDK env vars
    if [ -n "${ANDROID_NDK_HOME:-}" ] && [ -d "${ANDROID_NDK_HOME}" ]; then
        echo "${ANDROID_NDK_HOME}"
        return 0
    fi

    if [ -n "${ANDROID_NDK:-}" ] && [ -d "${ANDROID_NDK}" ]; then
        echo "${ANDROID_NDK}"
        return 0
    fi

    return 1
}

# Function to print error and exit
die() {
    echo -e "${RED}ERROR:${NC} $*" >&2
    exit 1
}

# Function to print success message
success() {
    echo -e "${GREEN}✓${NC} $*"
}

# Function to print warning
warn() {
    echo -e "${YELLOW}WARNING:${NC} $*"
}

# Function to print info
info() {
    echo "ℹ $*"
}

# Function to check required tools
check_tools() {
    info "Checking required tools..."

    local required_tools=(
        "cmake"
        "git"
        "ninja"
    )

    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            die "Required tool '${tool}' not found. Please install it."
        fi
        success "Found: ${tool}"
    done

    # Check NDK
    NDK_PATH=$(find_ndk) || die "Android NDK not found. Please install NDK r26b or later."
    success "Found NDK: ${NDK_PATH}"

    # Check NDK version
    if [ -f "${NDK_PATH}/source.properties" ]; then
        local ndk_version=$(grep "^Pkg.Revision" "${NDK_PATH}/source.properties" | cut -d= -f2)
        info "NDK Version: ${ndk_version}"
    fi
}

# Function to clone llama.cpp
clone_llamacpp() {
    info "Checking llama.cpp repository..."

    if [ -d "${LLAMACPP_DIR}" ]; then
        if [ -d "${LLAMACPP_DIR}/.git" ]; then
            info "llama.cpp repository exists, updating..."
            cd "${LLAMACPP_DIR}"
            git fetch --tags
            success "llama.cpp updated"
            cd "${SCRIPT_DIR}"
        else
            die "Directory exists but is not a git repository: ${LLAMACPP_DIR}"
        fi
    else
        info "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMACPP_DIR}"
        success "llama.cpp cloned"
    fi
}

# Function to build llama.cpp
build_llamacpp() {
    info "Configuring CMake for Android ARM64..."

    # Clean build directory
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # CMake toolchain file from NDK
    local TOOLCHAIN="${NDK_PATH}/build/cmake/android.toolchain.cmake"

    if [ ! -f "${TOOLCHAIN}" ]; then
        die "CMake toolchain not found: ${TOOLCHAIN}"
    fi

    # CMake configuration for Android ARM64 with Vulkan
    cmake -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_PLATFORM="android-${API_LEVEL}" \
        -DANDROID_STL="c++_shared" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}" \
        -DLLAMA_VULKAN=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CCACHE=OFF \
        -DBUILD_SHARED_LIBS=ON \
        ../

    success "CMake configuration complete"

    info "Building llama.cpp..."
    cmake --build . --config Release --parallel "$(nproc)"

    success "Build complete"
}

# Function to copy outputs
copy_outputs() {
    info "Copying build outputs..."

    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/bin"
    mkdir -p "${OUTPUT_DIR}/lib"
    mkdir -p "${OUTPUT_DIR}/include"

    # Copy shared library
    if [ -f "${BUILD_DIR}/libllama.so" ]; then
        cp "${BUILD_DIR}/libllama.so" "${OUTPUT_DIR}/lib/"
        success "Copied: libllama.so"
    else
        warn "libllama.so not found"
    fi

    # Copy finetune binary
    if [ -f "${BUILD_DIR}/bin/finetune" ]; then
        cp "${BUILD_DIR}/bin/finetune" "${OUTPUT_DIR}/bin/"
        success "Copied: finetune binary"
    else
        warn "finetune binary not found"
    fi

    # Copy main binary
    if [ -f "${BUILD_DIR}/bin/main" ]; then
        cp "${BUILD_DIR}/bin/main" "${OUTPUT_DIR}/bin/"
        success "Copied: main binary"
    fi

    # Copy other relevant binaries
    local binaries=("quantize" "export-lora" "train-text-from-scratch")
    for bin in "${binaries[@]}"; do
        if [ -f "${BUILD_DIR}/bin/${bin}" ]; then
            cp "${BUILD_DIR}/bin/${bin}" "${OUTPUT_DIR}/bin/"
            success "Copied: ${bin}"
        fi
    done

    # Copy headers
    if [ -d "${LLAMACPP_DIR}/include" ]; then
        cp -r "${LLAMACPP_DIR}/include/"* "${OUTPUT_DIR}/include/"
        success "Copied: headers"
    fi

    # Copy ggml headers
    if [ -f "${LLAMACPP_DIR}/ggml/include/ggml.h" ]; then
        mkdir -p "${OUTPUT_DIR}/include/ggml"
        cp "${LLAMACPP_DIR}/ggml/include/ggml.h" "${OUTPUT_DIR}/include/ggml/"
        success "Copied: ggml headers"
    fi
}

# Function to print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Build Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Built files:"
    [ -f "${OUTPUT_DIR}/lib/libllama.so" ] && echo "  ✓ lib/libllama.so"
    [ -f "${OUTPUT_DIR}/bin/finetune" ] && echo "  ✓ bin/finetune (LoRA training)"
    [ -f "${OUTPUT_DIR}/bin/main" ] && echo "  ✓ bin/main (inference)"
    [ -f "${OUTPUT_DIR}/bin/quantize" ] && echo "  ✓ bin/quantize"
    [ -f "${OUTPUT_DIR}/bin/export-lora" ] && echo "  ✓ bin/export-lora"
    echo ""
    echo "Configuration:"
    echo "  Target: ARM64-v8a (Android)"
    echo "  API Level: ${API_LEVEL}"
    echo "  Vulkan: Enabled (for Mali-G57)"
    echo "  LoRA: Enabled (finetune example)"
    echo "  GGUF: Supported"
    echo ""
    echo "To install on UBS1 phone:"
    echo "  adb push ${OUTPUT_DIR}/lib/libllama.so /data/local/tmp/"
    echo "  adb push ${OUTPUT_DIR}/bin/* /data/local/tmp/"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "QVAC Android ARM64 Build Script"
    echo "=========================================="
    echo ""

    check_tools
    clone_llamacpp
    build_llamacpp
    copy_outputs
    print_summary
}

# Run main function
main "$@"
