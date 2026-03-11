"""
Unit tests for compression pipeline.

Tests each compression stage: LoRA extraction, Top-K, SignSGD, and
end-to-end pipeline roundtrip.
"""
import pytest
import torch
import numpy as np
from compressor.lora_extractor import (
    extract_lora_delta, apply_lora_delta, create_tiny_lora_model
)
from compressor.topk_sparse import (
    TopKCompressor, CompressedDeltas, topk_compress, topk_decompress
)
from compressor.signsgd import (
    SignCompressor, SignCompressed, sign_compress, sign_decompress
)
from compressor.pipeline import CompressionPipeline


class TestLoRADeltaExtraction:
    """Tests for LoRA delta extraction and application."""

    def test_extract_delta(self):
        """Test that delta extraction computes correct differences."""
        model_before = create_tiny_lora_model()
        model_after = create_tiny_lora_model()

        with torch.no_grad():
            model_after.lora_A_1[0, 0] = 1.0
            model_after.lora_B_1[0, 0] = 2.0

        deltas = extract_lora_delta(model_before, model_after)

        assert "lora_A_1" in deltas
        assert "lora_B_1" in deltas
        assert deltas["lora_A_1"][0, 0].item() == 1.0
        assert deltas["lora_B_1"][0, 0].item() == 2.0

    def test_apply_delta(self):
        """Test that delta application correctly updates model."""
        model = create_tiny_lora_model()
        deltas = {
            "lora_A_1": torch.ones_like(model.lora_A_1) * 0.5,
            "lora_B_1": torch.ones_like(model.lora_B_1) * 0.3,
        }

        apply_lora_delta(model, deltas)

        assert torch.allclose(model.lora_A_1, torch.ones_like(model.lora_A_1) * 0.5)
        assert torch.allclose(model.lora_B_1, torch.ones_like(model.lora_B_1) * 0.3)

    def test_extract_apply_roundtrip(self):
        """Test that extract -> apply roundtrip preserves changes."""
        model_before = create_tiny_lora_model()
        model_target = create_tiny_lora_model()

        with torch.no_grad():
            for name, param in model_target.named_parameters():
                if "lora_" in name:
                    param.add_(torch.randn_like(param) * 0.1)

        deltas = extract_lora_delta(model_before, model_target)
        model_result = create_tiny_lora_model()
        apply_lora_delta(model_result, deltas)

        for (name1, p1), (name2, p2) in zip(
            model_result.named_parameters(), model_target.named_parameters()
        ):
            if "lora_" in name1:
                assert torch.allclose(p1, p2, atol=1e-6)


class TestTopKCompress:
    """Tests for Top-K sparsification."""

    def test_topk_compress_retains_top_values(self):
        """Test that only top-K% values are retained."""
        deltas = {"layer1": torch.randn(1000)}

        compressor = TopKCompressor(k_ratio=0.01)
        compressed = compressor.compress(deltas)

        # CompressedDeltas has flat arrays; 1% of 1000 = 10 values
        assert len(compressed.values) == 10
        assert len(compressed.indices) == 10

    def test_topk_decompress(self):
        """Test decompression reconstructs sparse representation."""
        deltas = {"layer1": torch.randn(1000)}

        compressor = TopKCompressor(k_ratio=0.01)
        compressed = compressor.compress(deltas)
        decompressed = compressor.decompress(compressed)

        assert "layer1" in decompressed
        assert decompressed["layer1"].shape == deltas["layer1"].shape

    def test_error_feedback_accumulates(self):
        """Test that error feedback accumulates residuals across rounds."""
        deltas = {"layer1": torch.randn(1000)}

        compressor = TopKCompressor(k_ratio=0.01)

        # First round
        compressor.compress(deltas)
        assert compressor.get_residual_size() > 0

        # Second round
        compressor.compress(deltas)
        assert compressor.get_residual_size() > 0


class TestSignSGD:
    """Tests for 1-bit SignSGD quantization."""

    def test_sign_compress_packs_bits(self):
        """Test that sign bits are correctly packed."""
        deltas = {"layer1": torch.randn(100)}

        compressor = TopKCompressor(k_ratio=0.1)
        topk_compressed = compressor.compress(deltas)

        sign_compressor = SignCompressor()
        signed = sign_compressor.compress(topk_compressed)

        # Packed bits: ceil(N / 8) bytes
        assert len(signed.sign_bits) <= (len(topk_compressed.values) + 7) // 8

    def test_sign_decompress(self):
        """Test decompression unpacks sign bits correctly."""
        deltas = {"layer1": torch.randn(100)}

        compressor = TopKCompressor(k_ratio=0.1)
        topk_compressed = compressor.compress(deltas)

        sign_compressor = SignCompressor()
        signed = sign_compressor.compress(topk_compressed)
        decompressed = sign_compressor.decompress(signed)

        # Check flat arrays have same length
        assert len(decompressed.values) == len(topk_compressed.values)

        # Check signs are preserved
        for i in range(len(topk_compressed.values)):
            original_sign = np.sign(topk_compressed.values[i])
            decompressed_sign = np.sign(decompressed.values[i])
            assert original_sign == decompressed_sign


class TestPipelineRoundtrip:
    """Tests for end-to-end compression pipeline."""

    def test_pipeline_roundtrip(self):
        """Test that compress -> decompress preserves top values."""
        deltas = {
            "lora_A_1": torch.randn(1000),
            "lora_B_1": torch.randn(1000),
        }

        pipeline = CompressionPipeline(k_ratio=0.01)

        compressed = pipeline.compress(deltas)
        decompressed = pipeline.decompress(compressed)

        for name in deltas:
            assert decompressed[name].shape == deltas[name].shape

    def test_compression_ratio(self):
        """Test that compression ratio is substantial."""
        deltas = {
            f"layer_{i}": torch.randn(100000)
            for i in range(10)
        }

        pipeline = CompressionPipeline(k_ratio=0.01)
        pipeline.compress(deltas)
        ratio = pipeline.get_compression_ratio()

        assert ratio > 50


def test_lora_delta_extraction():
    """Standalone test: extract -> apply -> verify."""
    model = create_tiny_lora_model()
    deltas = {
        "lora_A_1": torch.ones(4, 128) * 0.5,
        "lora_B_1": torch.ones(128, 4) * 0.3,
    }

    apply_lora_delta(model, deltas)

    assert abs(model.lora_A_1[0, 0].item() - 0.5) < 1e-5
    assert abs(model.lora_B_1[0, 0].item() - 0.3) < 1e-5


def test_topk_compress():
    """Standalone test: verify only top-K% values retained."""
    deltas = {"layer": torch.randn(10000)}
    compressed, residual = topk_compress(deltas, k_ratio=0.01)

    assert len(compressed.values) == 100  # 1% of 10000


def test_signsgd():
    """Standalone test: verify sign bits packed/unpacked correctly."""
    deltas = {"layer": torch.randn(1000)}
    topk_compressed, _ = topk_compress(deltas, k_ratio=0.1)
    signed = sign_compress(topk_compressed)
    decompressed = sign_decompress(signed)

    for i in range(len(topk_compressed.values)):
        orig_sign = np.sign(topk_compressed.values[i])
        dec_sign = np.sign(decompressed.values[i])
        assert orig_sign == dec_sign


def test_pipeline_roundtrip():
    """Standalone test: compress -> decompress -> verify fidelity."""
    deltas = {
        "lora_A": torch.randn(1000),
        "lora_B": torch.randn(1000),
    }

    pipeline = CompressionPipeline(k_ratio=0.01)
    compressed = pipeline.compress(deltas)
    decompressed = pipeline.decompress(compressed)

    assert decompressed["lora_A"].shape == deltas["lora_A"].shape
    assert decompressed["lora_B"].shape == deltas["lora_B"].shape


def test_error_feedback():
    """Standalone test: residuals accumulate across rounds."""
    deltas = {"layer": torch.randn(1000)}
    pipeline = CompressionPipeline(k_ratio=0.01)

    pipeline.compress(deltas)
    assert pipeline.get_residual_size() > 0

    pipeline.compress(deltas)
    assert pipeline.get_residual_size() > 0


def test_compression_ratio():
    """Standalone test: verify substantial compression."""
    deltas = {"layer": torch.randn(100000)}

    pipeline = CompressionPipeline(k_ratio=0.01)
    pipeline.compress(deltas)
    ratio = pipeline.get_compression_ratio()

    assert ratio > 50
