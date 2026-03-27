"""
Integration test for real Qwen2.5-1.5B-Instruct model with compression pipeline.

This test downloads the Qwen2.5-1.5B-Instruct model (approximately 3GB),
applies LoRA adapters, and validates the full compression pipeline with
real model deltas (not mock data).

The test verifies:
1. Model loading with INT4 quantization + LoRA
2. Forward pass with sample input
3. LoRA delta extraction
4. Compression with expected dimensions
5. Decompression and reapplication
6. Output difference verification (< 1e-3)
7. Compression ratio and size logging
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen25_loader import load_qwen25_with_lora, count_trainable_parameters
from models.quantization import print_memory_summary, get_quantization_memory_savings
from compressor.lora_extractor import extract_lora_delta, apply_lora_delta
from compressor.pipeline import CompressionPipeline


@pytest.mark.integration
@pytest.mark.slow
class TestRealModelCompression:
    """Integration tests for real Qwen2.5 model with compression."""

    @pytest.fixture
    def model_name(self):
        """The HuggingFace model identifier."""
        return "Qwen/Qwen2.5-1.5B-Instruct"

    @pytest.fixture
    def lora_config(self):
        """LoRA configuration matching the spec."""
        return {
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
        }

    @pytest.fixture
    def sample_input(self):
        """Sample input for model forward pass."""
        return torch.randint(0, 1000, (1, 32))  # Batch size 1, sequence length 32

    @pytest.fixture
    def pipeline_config(self):
        """Compression pipeline configuration."""
        return {"k_ratio": 0.01}

    def test_load_model_with_lora(self, model_name, lora_config):
        """
        Test loading Qwen2.5-1.5B with INT4 quantization and LoRA.

        Verifies:
        - Model loads successfully
        - INT4 quantization is applied
        - LoRA adapters are added
        - Correct number of trainable parameters
        """
        print("\n" + "=" * 60)
        print("TEST: Loading Qwen2.5-1.5B with INT4 + LoRA")
        print("=" * 60)

        # Load model with INT4 quantization and LoRA
        model, tokenizer = load_qwen25_with_lora(
            model_name=model_name,
            quantization="INT4",
            **lora_config
        )

        # Verify model loaded
        assert model is not None
        assert tokenizer is not None

        # Count parameters
        trainable, total = count_trainable_parameters(model)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters (LoRA): {trainable:,}")
        print(f"Trainable percentage: {100 * trainable / total:.2f}%")

        # For LoRA rank=4, we expect a small percentage of trainable params
        # The exact percentage depends on the model architecture
        assert trainable > 0, "Expected some trainable parameters with LoRA"
        assert trainable < total * 0.05, "Trainable params should be < 5% of total for LoRA"

        # Print memory summary
        print_memory_summary(model, "Qwen2.5-1.5B + INT4 + LoRA")

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_forward_pass(self, model_name, lora_config, sample_input):
        """Test forward pass with sample input."""
        print("\n" + "=" * 60)
        print("TEST: Forward pass with sample input")
        print("=" * 60)

        model, _ = load_qwen25_with_lora(
            model_name=model_name,
            quantization="INT4",
            **lora_config
        )
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        # Verify output shape
        batch_size, seq_len = sample_input.shape
        vocab_size = output.logits.shape[-1]
        assert output.logits.shape == (batch_size, seq_len, vocab_size)

        print(f"Input shape: {sample_input.shape}")
        print(f"Output logits shape: {output.logits.shape}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_extract_lora_deltas(self, model_name, lora_config):
        """
        Test extracting LoRA deltas between model states.

        Verifies:
        - Delta extraction works with real model
        - Correct number of deltas extracted
        - Delta dimensions match LoRA config
        """
        print("\n" + "=" * 60)
        print("TEST: Extracting LoRA deltas")
        print("=" * 60)

        # Load base model
        model, _ = load_qwen25_with_lora(
            model_name=model_name,
            quantization="INT4",
            **lora_config
        )
        model.eval()

        # Save initial state
        state_before = model.state_dict()

        # Simulate training by perturbing LoRA weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.add_(torch.randn_like(param) * 0.01)

        # Extract deltas
        deltas = extract_lora_delta(state_before, model)

        print(f"Number of delta tensors: {len(deltas)}")

        # Verify deltas extracted
        assert len(deltas) > 0, "Expected to extract some LoRA deltas"

        # Expected: 28 layers * 4 projections * 2 matrices (A and B) = 224 deltas
        # But we only get deltas for parameters that changed
        expected_deltas_per_layer = 8  # 4 projections * 2 matrices
        num_layers = 28  # Qwen2.5-1.5B has 28 layers
        expected_min = num_layers * 2  # At least 2 deltas per layer
        expected_max = num_layers * expected_deltas_per_layer

        print(f"Expected delta range: {expected_min} - {expected_max}")
        assert expected_min <= len(deltas) <= expected_max, \
            f"Unexpected number of deltas: {len(deltas)}"

        # Verify delta shapes match LoRA rank
        for key, delta in deltas.items():
            if "lora_A" in key:
                # lora_A shape: (rank, hidden_dim)
                assert delta.shape[0] == lora_config["lora_rank"], \
                    f"LoRA A rank mismatch for {key}"
            elif "lora_B" in key:
                # lora_B shape: (hidden_dim, rank)
                assert delta.shape[-1] == lora_config["lora_rank"], \
                    f"LoRA B rank mismatch for {key}"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_compress_decompress_roundtrip(
        self,
        model_name,
        lora_config,
        sample_input,
        pipeline_config
    ):
        """
        Test full compression roundtrip: extract -> compress -> decompress -> apply.

        Verifies:
        - Compression produces expected sizes
        - Decompression reconstructs deltas
        - Reapplying deltas produces similar outputs
        - Output difference < 1e-3 tolerance
        - Compression ratio is meaningful
        """
        print("\n" + "=" * 60)
        print("TEST: Compression roundtrip")
        print("=" * 60)

        # Load model
        model, _ = load_qwen25_with_lora(
            model_name=model_name,
            quantization="INT4",
            **lora_config
        )
        model.eval()

        # Get initial output
        with torch.no_grad():
            output_before = model(sample_input).logits

        # Save initial state
        state_before = model.state_dict()

        # Simulate training by perturbing LoRA weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.add_(torch.randn_like(param) * 0.01)

        # Get output after perturbation
        with torch.no_grad():
            output_after = model(sample_input).logits

        # Extract deltas
        deltas = extract_lora_delta(state_before, model)
        print(f"Extracted {len(deltas)} LoRA deltas")

        # Calculate original size
        original_size = sum(
            delta.numel() * delta.element_size()
            for delta in deltas.values()
        )
        print(f"Original delta size: {original_size / (1024 ** 2):.2f} MB")

        # Expected: 28 layers * 4 projections * rank=4 * hidden=1536
        # Hidden dimension for Qwen2.5-1.5B is 1536
        expected_params = 28 * 4 * 4 * 1536 * 4  # *4 for float32
        print(f"Expected size: {expected_params / (1024 ** 2):.2f} MB")

        # Compress deltas
        pipeline = CompressionPipeline(**pipeline_config)
        compressed = pipeline.compress(deltas, original_size=original_size)
        print(f"Compressed size: {len(compressed) / (1024 ** 2):.4f} MB")

        # Get compression ratio
        ratio = pipeline.get_compression_ratio()
        print(f"Compression ratio: {ratio:.2f}x")

        # Verify meaningful compression
        assert ratio > 1.0, "Expected compression ratio > 1.0"

        # Decompress
        recovered = pipeline.decompress(compressed)
        print(f"Recovered {len(recovered)} LoRA deltas")

        # Reset model to before state
        for key, param in model.named_parameters():
            if key in state_before:
                param.data.copy_(state_before[key])

        # Apply recovered deltas
        apply_lora_delta(model, recovered)

        # Get output after recovery
        with torch.no_grad():
            output_recovered = model(sample_input).logits

        # Calculate difference
        diff = torch.abs(output_recovered - output_after).max().item()
        print(f"Max output difference: {diff:.6f}")

        # Verify output difference < 1e-3 (with tolerance for quantization)
        # Note: With Top-K + SignSGD, we may have larger differences due to aggressive compression
        tolerance = 1e-3
        assert diff < tolerance, \
            f"Output difference {diff} exceeds tolerance {tolerance}"

        # Print summary
        print("\n" + "-" * 60)
        print("COMPRESSION SUMMARY")
        print("-" * 60)
        print(f"Original size:     {original_size / (1024 ** 2):.2f} MB")
        print(f"Compressed size:   {len(compressed) / (1024 ** 2):.4f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Output diff:       {diff:.6f} (< {tolerance})")
        print("-" * 60)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_quantization_memory_savings(self):
        """Test that quantization provides expected memory savings."""
        print("\n" + "=" * 60)
        print("TEST: Quantization memory savings")
        print("=" * 60)

        # Calculate theoretical savings
        ratio_fp32_to_int4 = get_quantization_memory_savings(
            original_dtype=torch.float32,
            target_dtype="int4"
        )
        ratio_fp32_to_int8 = get_quantization_memory_savings(
            original_dtype=torch.float32,
            target_dtype="int8"
        )
        ratio_fp32_to_fp16 = get_quantization_memory_savings(
            original_dtype=torch.float32,
            target_dtype="fp16"
        )

        print(f"FP32 -> INT4:   {ratio_fp32_to_int4:.2f}x reduction")
        print(f"FP32 -> INT8:   {ratio_fp32_to_int8:.2f}x reduction")
        print(f"FP32 -> FP16:   {ratio_fp32_to_fp16:.2f}x reduction")

        # Verify expected ratios
        assert abs(ratio_fp32_to_int4 - 8.0) < 0.1, "FP32->INT4 should be ~8x"
        assert abs(ratio_fp32_to_int8 - 4.0) < 0.1, "FP32->INT8 should be ~4x"
        assert abs(ratio_fp32_to_fp16 - 2.0) < 0.1, "FP32->FP16 should be ~2x"


def main():
    """
    Run the integration tests.

    Usage:
        python -m tests.test_real_model

    Note: This test downloads ~3GB of model data and requires
    sufficient memory (> 8GB GPU recommended, or CPU with sufficient RAM).
    """
    import sys

    print("=" * 80)
    print("Qwen2.5-1.5B Real Model Integration Test")
    print("=" * 80)
    print("\nThis test will:")
    print("  1. Download Qwen/Qwen2.5-1.5B-Instruct (~3GB)")
    print("  2. Load with INT4 quantization")
    print("  3. Add LoRA adapters")
    print("  4. Run forward pass")
    print("  5. Extract LoRA deltas")
    print("  6. Compress with Top-K + SignSGD")
    print("  7. Decompress and verify roundtrip")
    print("\nEstimated time: 5-10 minutes (first run)")
    print("=" * 80)

    if "--skip-download" not in sys.argv:
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return

    # Run pytest with this file
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
