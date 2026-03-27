"""
SignSGD Quantization Module

This module implements 1-bit SignSGD quantization for extreme gradient compression.
Each value is replaced with its sign bit (±1) plus a global scale factor.
Sign bits are packed into uint8 arrays (8 values per byte) for efficient transmission.
"""

from typing import List
from dataclasses import dataclass
import torch
import numpy as np
import threading
from .topk_sparse import CompressedDeltas, TensorMetadata


@dataclass
class SignCompressed:
    """
    1-bit SignSGD compressed representation of deltas.

    Attributes:
        sign_bits: Packed uint8 array (8 values per byte)
        scale: Global scale factor (float32)
        indices: Original indices from Top-K sparse representation
        metadata: Metadata for reconstructing tensor shapes
    """
    sign_bits: np.ndarray  # uint8
    scale: np.float32
    indices: np.ndarray     # int32
    metadata: List[TensorMetadata]


def sign_compress(compressed: CompressedDeltas) -> SignCompressed:
    """
    Replace each value with its sign bit (1-bit) and compute global scale.

    This function implements 1-bit quantization where:
    - Each value is replaced by its sign (+1 or -1)
    - A global scale factor captures the average magnitude
    - Sign bits are packed into uint8 arrays (8 values per byte)

    Args:
        compressed: CompressedDeltas from Top-K sparsification

    Returns:
        SignCompressed: 1-bit quantized representation with packed sign bits

    Example:
        >>> # compressed is from topk_compress()
        >>> signed = sign_compress(compressed)
        >>> # signed.sign_bits is packed uint8 array
        >>> # signed.scale is global magnitude
    """
    if len(compressed.values) == 0:
        # Handle empty case
        return SignCompressed(
            sign_bits=np.array([], dtype=np.uint8),
            scale=np.float32(1.0),
            indices=compressed.indices.copy(),
            metadata=compressed.metadata.copy()
        )

    # Convert values to float32 for processing
    values = compressed.values.astype(np.float32)

    # Compute global scale factor (mean absolute value)
    abs_values = np.abs(values)
    scale = np.mean(abs_values) if len(abs_values) > 0 else np.float32(1.0)

    # Avoid division by zero
    if scale < 1e-8:
        scale = np.float32(1.0)

    # Compute sign bits (1 for positive, 0 for negative)
    signs = (values >= 0).astype(np.uint8)

    # Pack 8 sign bits into each byte
    n_values = len(signs)
    n_bytes = (n_values + 7) // 8
    packed_bits = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(n_values):
        byte_idx = i // 8
        bit_idx = i % 8
        if signs[i]:
            packed_bits[byte_idx] |= (1 << bit_idx)

    return SignCompressed(
        sign_bits=packed_bits,
        scale=scale,
        indices=compressed.indices.copy(),
        metadata=compressed.metadata.copy()
    )


def sign_decompress(signed: SignCompressed) -> CompressedDeltas:
    """
    Unpack sign bits and multiply by scale factor.

    Args:
        signed: SignCompressed object with 1-bit representation

    Returns:
        CompressedDeltas: Reconstructed with float16 values (±scale)

    Example:
        >>> compressed = sign_decompress(signed)
        >>> # compressed.values contains ±scale values
    """
    # Unpack sign bits
    n_values = len(signed.indices)
    signs = np.zeros(n_values, dtype=np.uint8)

    for i in range(n_values):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(signed.sign_bits):
            if signed.sign_bits[byte_idx] & (1 << bit_idx):
                signs[i] = 1
            else:
                signs[i] = 0

    # Convert signs to +1 or -1
    sign_values = np.where(signs > 0, np.float32(1.0), np.float32(-1.0))

    # Multiply by scale factor
    values = (sign_values * signed.scale).astype(np.float16)

    # Create CompressedDeltas object
    return CompressedDeltas(
        indices=signed.indices.copy(),
        values=values,
        metadata=signed.metadata.copy(),
        k_ratio=1.0  # k_ratio not used after sign compression
    )


class SignCompressor:
    """
    Thread-safe SignSGD compression.

    This class provides thread-safe 1-bit quantization operations
    for concurrent use in distributed training scenarios.
    """

    def __init__(self):
        """Initialize the SignSGD compressor with a lock for thread safety."""
        self._lock = threading.Lock()

    def compress(self, compressed: CompressedDeltas) -> SignCompressed:
        """
        Compress sparse deltas to 1-bit representation.

        Args:
            compressed: CompressedDeltas from Top-K sparsification

        Returns:
            SignCompressed: 1-bit quantized representation
        """
        with self._lock:
            return sign_compress(compressed)

    def decompress(self, signed: SignCompressed) -> CompressedDeltas:
        """
        Decompress 1-bit representation back to sparse format.

        Args:
            signed: SignCompressed object

        Returns:
            CompressedDeltas: Sparse representation with float16 values
        """
        with self._lock:
            return sign_decompress(signed)


def get_compression_ratio(
    original_size: int,
    sign_compressed: SignCompressed
) -> float:
    """
    Calculate the compression ratio achieved by the SignSGD compression.

    Args:
        original_size: Original size in bytes
        sign_compressed: SignCompressed object

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    # Calculate compressed size
    compressed_size = (
        len(sign_compressed.sign_bits) +  # packed sign bits (uint8)
        4 +                                 # scale factor (float32)
        len(sign_compressed.indices) * 4   # indices (int32)
    )

    # Estimate metadata size (simplified)
    for meta in sign_compressed.metadata:
        compressed_size += len(meta.name) + 8  # name + shape (2 ints) + start/end

    if compressed_size == 0:
        return float('inf')

    return original_size / compressed_size
