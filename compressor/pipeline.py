"""
Compression Pipeline Module

This module provides a unified compression pipeline that chains together:
1. LoRA delta extraction (from lora_extractor)
2. Top-K sparsification (from topk_sparse)
3. SignSGD quantization (from signsgd)
4. Serialization to bytes

The pipeline maintains error feedback state across compression rounds.
"""

import msgpack
import struct
from typing import Dict, Optional
import torch
import numpy as np
import threading

from .lora_extractor import extract_lora_delta, apply_lora_delta
from .topk_sparse import TopKCompressor, CompressedDeltas
from .signsgd import SignCompressor, SignCompressed, get_compression_ratio


class CompressionPipeline:
    """
    Complete gradient compression pipeline with serialization.

    This class provides a thread-safe end-to-end compression pipeline:
    - LoRA delta extraction
    - Top-K sparsification with error feedback
    - 1-bit SignSGD quantization
    - Serialization to bytes using msgpack

    Usage:
        >>> pipeline = CompressionPipeline(k_ratio=0.01)
        >>> # Compress LoRA deltas
        >>> compressed_bytes = pipeline.compress(deltas)
        >>> ratio = pipeline.get_compression_ratio()
        >>> # Decompress
        >>> reconstructed_deltas = pipeline.decompress(compressed_bytes)
    """

    def __init__(self, k_ratio: float = 0.01):
        """
        Initialize the compression pipeline.

        Args:
            k_ratio: Ratio of values to keep in Top-K sparsification
                    (0.01 = 1%, 0.1 = 10%, etc.)
        """
        self.k_ratio = k_ratio
        self.topk_compressor = TopKCompressor(k_ratio=k_ratio)
        self.sign_compressor = SignCompressor()
        self._original_size: int = 0
        self._compressed_size: int = 0
        self._lock = threading.Lock()

    def compress(
        self,
        deltas: Dict[str, torch.Tensor],
        original_size: Optional[int] = None
    ) -> bytes:
        """
        Compress LoRA deltas through the full pipeline.

        The compression chain:
        1. Extract LoRA deltas (already done by caller)
        2. Apply Top-K sparsification (keeps only top K% by magnitude)
        3. Apply SignSGD quantization (1-bit per value + global scale)
        4. Serialize to bytes using msgpack

        Args:
            deltas: Dictionary of parameter names to delta tensors
            original_size: Optional original size in bytes for compression ratio

        Returns:
            Serialized compressed data as bytes

        Example:
            >>> deltas = {"layer1.lora_A.weight": torch.randn(100, 100)}
            >>> compressed = pipeline.compress(deltas)
        """
        with self._lock:
            # Calculate original size if not provided
            if original_size is None:
                original_size = 0
                for tensor in deltas.values():
                    original_size += tensor.element_size() * tensor.nelement()
            self._original_size = original_size

            # Step 1: Top-K sparsification (keeps top K% by magnitude)
            sparse_deltas = self.topk_compressor.compress(deltas)

            # Step 2: SignSGD quantization (1-bit per value)
            sign_compressed = self.sign_compressor.compress(sparse_deltas)

            # Step 3: Serialize to bytes using msgpack
            compressed_bytes = self._serialize(sign_compressed)
            self._compressed_size = len(compressed_bytes)

            return compressed_bytes

    def decompress(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        Decompress bytes back to LoRA deltas.

        Reverse of compress():
        1. Deserialize from msgpack
        2. SignSGD decompression (unpack bits, apply scale)
        3. Top-K decompression (reconstruct sparse tensors)

        Args:
            data: Serialized compressed data from compress()

        Returns:
            Dictionary of parameter names to reconstructed delta tensors

        Example:
            >>> deltas = pipeline.decompress(compressed_bytes)
        """
        with self._lock:
            # Step 1: Deserialize from msgpack
            sign_compressed = self._deserialize(data)

            # Step 2: SignSGD decompression
            sparse_deltas = self.sign_compressor.decompress(sign_compressed)

            # Step 3: Top-K decompression
            deltas = self.topk_compressor.decompress(sparse_deltas)

            return deltas

    def get_compression_ratio(self) -> float:
        """
        Get the compression ratio achieved by the last compression.

        Returns:
            Compression ratio (original_size / compressed_size)
            Returns 0.0 if no compression has been performed
        """
        with self._lock:
            if self._compressed_size == 0:
                return 0.0
            return self._original_size / self._compressed_size

    def clear_residual(self) -> None:
        """
        Clear the error feedback accumulator for the Top-K compressor.

        This should be called when starting a new training round
        to reset the error feedback state.
        """
        with self._lock:
            self.topk_compressor.clear_residual()

    def get_residual_size(self) -> int:
        """
        Get the size of the error feedback accumulator in bytes.

        Returns:
            Size of residual tensors in bytes
        """
        with self._lock:
            return self.topk_compressor.get_residual_size()

    def _serialize(self, sign_compressed: SignCompressed) -> bytes:
        """
        Serialize SignCompressed object to bytes using msgpack.

        Args:
            sign_compressed: SignCompressed object to serialize

        Returns:
            Serialized bytes
        """
        # Convert metadata to serializable format
        metadata_list = []
        for meta in sign_compressed.metadata:
            metadata_list.append({
                'name': meta.name,
                'shape': list(meta.shape),
                'start_idx': meta.start_idx,
                'end_idx': meta.end_idx
            })

        # Create serializable dictionary
        data = {
            'sign_bits': sign_compressed.sign_bits.tobytes(),
            'scale': float(sign_compressed.scale),
            'indices': sign_compressed.indices.tobytes(),
            'metadata': metadata_list
        }

        # Serialize with msgpack
        return msgpack.packb(data, use_bin_type=True)

    def _deserialize(self, data: bytes) -> SignCompressed:
        """
        Deserialize bytes to SignCompressed object.

        Args:
            data: Serialized bytes from _serialize()

        Returns:
            SignCompressed object
        """
        # Deserialize with msgpack
        unpacked = msgpack.unpackb(data, raw=False)

        # Reconstruct metadata
        metadata_list = []
        for meta_dict in unpacked['metadata']:
            from .topk_sparse import TensorMetadata
            metadata_list.append(TensorMetadata(
                name=meta_dict['name'],
                shape=tuple(meta_dict['shape']),
                start_idx=meta_dict['start_idx'],
                end_idx=meta_dict['end_idx']
            ))

        # Reconstruct arrays
        sign_bits = np.frombuffer(unpacked['sign_bits'], dtype=np.uint8)
        indices = np.frombuffer(unpacked['indices'], dtype=np.int32)

        return SignCompressed(
            sign_bits=sign_bits,
            scale=np.float32(unpacked['scale']),
            indices=indices,
            metadata=metadata_list
        )


def extract_compress(
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
    pipeline: CompressionPipeline
) -> bytes:
    """
    Convenience function to extract LoRA deltas and compress them.

    Args:
        model_before: Model before training
        model_after: Model after training
        pipeline: CompressionPipeline instance

    Returns:
        Compressed bytes

    Example:
        >>> pipeline = CompressionPipeline(k_ratio=0.01)
        >>> compressed = extract_compress(base_model, trained_model, pipeline)
    """
    deltas = extract_lora_delta(model_before, model_after)
    return pipeline.compress(deltas)


def decompress_apply(
    data: bytes,
    model: torch.nn.Module,
    pipeline: CompressionPipeline
) -> None:
    """
    Convenience function to decompress and apply LoRA deltas.

    Args:
        data: Compressed bytes
        model: Model to apply deltas to
        pipeline: CompressionPipeline instance

    Example:
        >>> decompress_apply(compressed, server_model, pipeline)
    """
    deltas = pipeline.decompress(data)
    apply_lora_delta(model, deltas)
