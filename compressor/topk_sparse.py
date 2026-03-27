"""
Top-K Sparsification Module

This module implements Top-K sparsification for gradient compression.
It keeps only the top-K% values by magnitude, storing indices and values
in a compact format with error feedback for accumulated residuals.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np
import threading


@dataclass
class TensorMetadata:
    """Metadata for reconstructing tensor shapes from sparse representation."""
    name: str
    shape: Tuple[int, ...]
    start_idx: int  # Starting index in the flattened sparse array
    end_idx: int    # Ending index in the flattened sparse array


@dataclass
class CompressedDeltas:
    """
    Sparse representation of compressed deltas using Top-K sparsification.

    Attributes:
        indices: int32 array of indices into the original flattened tensors
        values: float16 array of the top-K values
        metadata: List of TensorMetadata objects for reconstruction
        k_ratio: The ratio of values kept (e.g., 0.01 for 1%)
    """
    indices: np.ndarray  # int32
    values: np.ndarray   # float16
    metadata: List[TensorMetadata]
    k_ratio: float


def topk_compress(
    deltas: Dict[str, torch.Tensor],
    k_ratio: float = 0.01,
    residual: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[CompressedDeltas, Dict[str, torch.Tensor]]:
    """
    Compress LoRA deltas using Top-K sparsification.

    This function flattens all deltas and keeps only the top-K% values by magnitude.
    It supports error feedback by accumulating residuals from previous rounds.

    Args:
        deltas: Dictionary of parameter names to delta tensors
        k_ratio: Ratio of values to keep (0.01 = 1%, 0.1 = 10%)
        residual: Optional error feedback accumulator from previous round

    Returns:
        Tuple of (CompressedDeltas, new_residual):
        - CompressedDeltas: Sparse representation with indices and values
        - new_residual: Updated error accumulator for next round

    Example:
        >>> deltas = {"layer1.lora_A.weight": torch.randn(100, 100)}
        >>> compressed, residual = topk_compress(deltas, k_ratio=0.01)
    """
    if residual is None:
        residual = {}

    # Flatten all deltas and track metadata
    all_indices = []
    all_values = []
    metadata = []
    new_residual = {}

    for name, delta in deltas.items():
        # Add error feedback from previous round
        if name in residual:
            delta = delta + residual[name]

        # Flatten and compute absolute values for top-k selection
        flat_delta = delta.flatten()
        abs_values = torch.abs(flat_delta)

        # Determine k (number of values to keep)
        k = max(1, int(len(flat_delta) * k_ratio))

        # Get top-k indices
        _, topk_indices = torch.topk(abs_values, k)

        # Extract top-k values
        topk_values = flat_delta[topk_indices]

        # Store error feedback (values not in top-k)
        error_mask = torch.ones_like(flat_delta, dtype=torch.bool)
        error_mask[topk_indices] = False
        error_delta = torch.where(error_mask, flat_delta, torch.zeros_like(flat_delta))
        new_residual[name] = error_delta.reshape(delta.shape)

        # Keep indices as-is (local to each tensor)
        # No offset needed since each tensor is reconstructed independently
        # Calculate start_idx as total length before appending
        start_idx = sum(len(arr) for arr in all_indices)
        all_indices.append(topk_indices.cpu().numpy())
        all_values.append(topk_values.cpu().numpy())
        # Calculate end_idx as total length after appending
        end_idx = sum(len(arr) for arr in all_indices)
        metadata.append(TensorMetadata(
            name=name,
            shape=tuple(delta.shape),
            start_idx=start_idx,
            end_idx=end_idx
        ))

    # Concatenate all indices and values
    if all_indices:
        indices = np.concatenate(all_indices).astype(np.int32)
        values = np.concatenate(all_values).astype(np.float16)
    else:
        indices = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float16)

    compressed = CompressedDeltas(
        indices=indices,
        values=values,
        metadata=metadata,
        k_ratio=k_ratio
    )

    return compressed, new_residual


def topk_decompress(
    compressed: CompressedDeltas,
    output_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct full-size tensors from sparse Top-K representation.

    Args:
        compressed: CompressedDeltas object with sparse representation
        output_shape: Optional shape hint (for validation, not used)

    Returns:
        Dictionary of parameter names to reconstructed tensors

    Example:
        >>> deltas = topk_decompress(compressed)
    """
    if len(compressed.indices) == 0:
        return {}

    # Extract tensors based on metadata
    deltas = {}
    for meta in compressed.metadata:
        # Extract the slice of indices and values for this tensor
        indices_slice = compressed.indices[meta.start_idx:meta.end_idx]
        values_slice = compressed.values[meta.start_idx:meta.end_idx]

        # Create the reconstructed tensor (all zeros)
        reconstructed = torch.zeros(meta.shape, dtype=torch.float16)
        flat_tensor = reconstructed.flatten()

        # The indices during compression were not offset - they are local to each tensor
        # So we can use them directly to index into the flattened tensor
        for idx, value in zip(indices_slice, values_slice):
            local_idx = int(idx)  # Indices are already local to this tensor
            if 0 <= local_idx < len(flat_tensor):
                flat_tensor[local_idx] = float(value)

        deltas[meta.name] = reconstructed

    return deltas


class TopKCompressor:
    """
    Thread-safe Top-K compression with persistent error feedback state.

    This class maintains error feedback state across compression rounds
    and provides thread-safe operations for concurrent use.
    """

    def __init__(self, k_ratio: float = 0.01):
        """
        Initialize the Top-K compressor.

        Args:
            k_ratio: Ratio of values to keep (0.01 = 1%)
        """
        self.k_ratio = k_ratio
        self._residual: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def compress(
        self,
        deltas: Dict[str, torch.Tensor]
    ) -> CompressedDeltas:
        """
        Compress deltas with error feedback.

        Args:
            deltas: Dictionary of parameter names to delta tensors

        Returns:
            CompressedDeltas: Sparse representation
        """
        with self._lock:
            compressed, self._residual = topk_compress(
                deltas,
                k_ratio=self.k_ratio,
                residual=self._residual
            )
            return compressed

    def decompress(self, compressed: CompressedDeltas) -> Dict[str, torch.Tensor]:
        """
        Decompress sparse representation back to tensors.

        Args:
            compressed: CompressedDeltas object

        Returns:
            Dictionary of parameter names to reconstructed tensors
        """
        with self._lock:
            return topk_decompress(compressed)

    def clear_residual(self) -> None:
        """Clear the error feedback accumulator."""
        with self._lock:
            self._residual = {}

    def get_residual_size(self) -> int:
        """
        Get the size of the error feedback accumulator in bytes.

        Returns:
            Size of residual tensors in bytes
        """
        with self._lock:
            size = 0
            for tensor in self._residual.values():
                size += tensor.element_size() * tensor.nelement()
            return size
