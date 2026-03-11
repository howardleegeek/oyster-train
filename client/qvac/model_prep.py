#!/usr/bin/env python3
"""
Model preparation script for QVAC Fabric LLM on Android.
Downloads Qwen2.5-1.5B-Instruct from HuggingFace, converts to GGUF format (INT4 quantization),
and sets up LoRA adapter configuration.
Outputs to client/qvac/models/
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Try to import required packages
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    import torch
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Please install required packages:")
    print("pip install transformers huggingface_hub torch")
    sys.exit(1)

def main():
    print("=== QVAC Fabric LLM Model Preparation ===")
    
    # Define paths
    script_dir = Path(__file__).parent.absolute()
    models_dir = script_dir / "models"
    llama_cpp_dir = script_dir / "llama.cpp"
    
    # Create models directory
    models_dir.mkdir(exist_ok=True)
    
    # Check if llama.cpp exists
    if not llama_cpp_dir.exists():
        print(f"Error: llama.cpp directory not found at {llama_cpp_dir}")
        print("Please ensure the QVAC Fabric LLM (llama.cpp fork) is present in client/qvac/llama.cpp")
        sys.exit(1)
    
    # Check for convert.py in llama.cpp
    convert_script = llama_cpp_dir / "convert.py"
    if not convert_script.exists():
        print(f"Error: convert.py not found in {llama_cpp_dir}")
        print("The llama.cpp directory may be incomplete or incorrect.")
        sys.exit(1)
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    temp_model_dir = script_dir / "temp_model"
    gguf_output_dir = models_dir
    
    try:
        # Step 1: Download model from HuggingFace
        print(f"Downloading model {model_name} from HuggingFace...")
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)
        temp_model_dir.mkdir()
        
        # Download using snapshot_download for full model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(temp_model_dir / "cache"),
            local_dir=str(temp_model_dir / "model"),
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to: {model_path}")
        
        # Step 2: Convert to GGUF format using llama.cpp's convert.py
        print("Converting model to GGUF format (INT4 quantization)...")
        gguf_model_path = gguf_output_dir / "qwen2.5-1.5b-instruct-q4_0.gguf"
        
        # Run the conversion script
        cmd = [
            sys.executable,
            str(convert_script),
            model_path,
            "--outtype", "q4_0",
            "--outfile", str(gguf_model_path)
        ]
        
        print(f"Running conversion command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error during conversion:")
            print(result.stderr)
            sys.exit(1)
        
        print(f"Conversion successful! Model saved to: {gguf_model_path}")
        
        # Step 3: Create LoRA configuration file (for reference)
        lora_config = {
            "model_type": "qwen2",
            "r": 4,
            "alpha": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        import json
        lora_config_path = models_dir / "lora_config.json"
        with open(lora_config_path, 'w') as f:
            json.dump(lora_config, f, indent=2)
        print(f"LoRA configuration saved to: {lora_config_path}")
        
        # Step 4: Estimate memory usage
        print("\n=== Memory Usage Estimation (6GB Device) ===")
        print("Model: Qwen2.5-1.5B-Instruct (INT4 quantized)")
        print("Parameter count: ~1.5 billion")
        print("Quantization: 4-bit")
        print(f"Model size on disk: ~{gguf_model_path.stat().st_size / (1024**2):.1f} MB")
        print("Estimated runtime memory usage:")
        print("  - Model weights: ~0.5 GB")
        print("  - KV cache (512 tokens): ~0.3 GB")
        print("  - Overhead and working memory: ~0.5 GB")
        print("  - Total estimated: ~1.3 GB")
        print("Suitable for devices with 3GB+ RAM (targeting 6GB devices)")
        
        # Step 5: Clean up temporary files
        print("\nCleaning up temporary files...")
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)
        
        print("\n=== Model Preparation Complete ===")
        print(f"GGUF model: {gguf_model_path}")
        print(f"LoRA config: {lora_config_path}")
        print("You can now run the build script: ./build_android.sh")
        
    except Exception as e:
        print(f"Error during model preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()