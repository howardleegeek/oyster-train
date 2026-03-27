"""
Oyster-Train Compression Package

This package provides a 3-layer gradient compression pipeline for federated learning
with LoRA-fine-tuned models.

The pipeline achieves 300x+ compression by combining:
1. LoRA delta extraction (extracts only changed parameters)
2. Top-K sparsification (keeps only top 1% by magnitude)
3. 1-bit SignSGD quantization (sign bits + global scale)

Typical usage:
    >>> from compressor import CompressionPipeline
    >>> pipeline = CompressionPipeline(k_ratio=0.01)
    >>> compressed = pipeline.compress(deltas)
    >>> ratio = pipeline.get_compression_ratio()
    >>> reconstructed = pipeline.decompress(compressed)
"""

from .lora_extractor import (
    extract_lora_delta,
    apply_lora_delta,
    LoRAExtractor
)

from .topk_sparse import (
    topk_compress,
    topk_decompress,
    CompressedDeltas,
    TensorMetadata,
    TopKCompressor
)

from .signsgd import (
    sign_compress,
    sign_decompress,
    SignCompressed,
    SignCompressor,
    get_compression_ratio
)

from .pipeline import (
    CompressionPipeline,
    extract_compress,
    decompress_apply
)

__all__ = [
    # LoRA extraction
    'extract_lora_delta',
    'apply_lora_delta',
    'LoRAExtractor',

    # Top-K sparsification
    'topk_compress',
    'topk_decompress',
    'CompressedDeltas',
    'TensorMetadata',
    'TopKCompressor',

    # SignSGD quantization
    'sign_compress',
    'sign_decompress',
    'SignCompressed',
    'SignCompressor',
    'get_compression_ratio',

    # Pipeline
    'CompressionPipeline',
    'extract_compress',
    'decompress_apply',
]

__version__ = '0.1.0'
