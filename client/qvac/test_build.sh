#!/bin/bash
# Verify build output exists and is ARM64 binary
# Check linked libraries (Vulkan, NEON)
# Print binary size

# Check if build directory exists
if [ ! -d "build" ]; then
  echo "Error: Build directory not found. Please run build_android.sh first."
  exit 1
fi

cd build

# Check for main binary (llama-cli or similar)
BINARY="llama-cli"
if [ ! -f "$BINARY" ]; then
  # Try alternative names
  if [ -f "llama" ]; then
    BINARY="llama"
  elif [ -f "main" ]; then
    BINARY="main"
  else
    echo "Error: No binary found in build directory"
    ls -la
    exit 1
  fi
fi

echo "Found binary: $BINARY"

# Check if it's an ARM64 binary using file command
if command -v file >/dev/null 2>&1; then
  FILE_OUTPUT=$(file "$BINARY")
  echo "File type: $FILE_OUTPUT"
  
  # Check for ARM64
  if echo "$FILE_OUTPUT" | grep -q "ARM aarch64"; then
    echo "✓ Binary is ARM64 (arm64-v8a) as expected"
  else
    echo "⚠ Warning: Binary may not be ARM64"
  fi
  
  # Check for dynamic linking info
  if echo "$FILE_OUTPUT" | grep -q "dynamically linked"; then
    echo "✓ Binary is dynamically linked"
  else
    echo "ℹ Binary is statically linked (common for Android builds)"
  fi
else
  echo "ℹ 'file' command not available, skipping binary type check"
fi

# Check binary size
SIZE=$(du -h "$BINARY" | cut -f1)
echo "Binary size: $SIZE"

# Check for Vulkan and NEON symbols (if we have readelf or nm)
if command -v readelf >/dev/null 2>&1; then
  echo ""
  echo "Checking for key symbols:"
  
  # Check for Vulkan symbols
  if readelf -s "$BINARY" | grep -q "vulkan"; then
    echo "✓ Vulkan symbols found"
  else
    echo "⚠ No Vulkan symbols found (may be stripped or not linked)"
  fi
  
  # Check for NEON-related functions (simplified check)
  if readelf -s "$BINARY" | grep -q -i "neon"; then
    echo "✓ NEON-related symbols found"
  else
    echo "ℹ No explicit NEON symbols found (may be in intrinsics)"
  fi
  
  # Check for LoRA training symbols
  if readelf -s "$BINARY" | grep -q "lora"; then
    echo "✓ LoRA training symbols found"
  else
    echo "ℹ No LoRA symbols found (may be stripped or not built)"
  fi
elif command -v nm >/dev/null 2>&1; then
  echo ""
  echo "Checking for key symbols (using nm):"
  
  # Check for Vulkan symbols
  if nm "$BINARY" | grep -q "vulkan"; then
    echo "✓ Vulkan symbols found"
  else
    echo "⚠ No Vulkan symbols found"
  fi
  
  # Check for NEON-related functions
  if nm "$BINARY" | grep -q -i "neon"; then
    echo "✓ NEON-related symbols found"
  else
    echo "ℹ No explicit NEON symbols found"
  fi
  
  # Check for LoRA training symbols
  if nm "$BINARY" | grep -q "lora"; then
    echo "✓ LoRA training symbols found"
  else
    echo "ℹ No LoRA symbols found"
  fi
else
  echo ""
  echo "ℹ Neither 'readelf' nor 'nm' available, skipping symbol checks"
fi

# Check if models directory exists and has content
if [ -d "../models" ] && [ "$(ls -A ../models)" ]; then
  echo ""
  echo "✓ Models directory exists and contains files:"
  ls -la ../models/
else
  echo ""
  echo "⚠ Models directory is empty or missing. Run model_prep.py to prepare models."
fi

echo ""
echo "Build verification complete."
cd ..