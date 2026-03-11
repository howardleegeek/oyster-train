"""
LoRA Delta Extraction Module

This module provides functionality to extract LoRA deltas between two model checkpoints
and apply aggregated deltas back to a model.

LoRA (Low-Rank Adaptation) parameters are stored in A and B matrices for each
transformer layer. This module computes the differences (deltas) between checkpoints.
"""

from typing import Dict
import torch
import threading


def extract_lora_delta(
    model_before: torch.nn.Module,
    model_after: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute the difference of LoRA A and B matrices between two model checkpoints.

    Args:
        model_before: The model state before training (base checkpoint)
        model_after: The model state after training (updated checkpoint)

    Returns:
        Dictionary mapping LoRA parameter names to their delta tensors.
        Keys follow the pattern: "base_model.model.model.layers.{i}.{proj}.lora_{A/B}.weight"

    Example:
        >>> deltas = extract_lora_delta(base_model, trained_model)
        >>> # deltas contains weight differences for all LoRA parameters
    """
    deltas = {}
    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    # Find all LoRA parameters
    for key in state_dict_after:
        # Check if this is a LoRA parameter (A or B matrix)
        if "lora_A" in key or "lora_B" in key:
            if key in state_dict_before:
                # Compute the difference
                delta = state_dict_after[key] - state_dict_before[key]
                deltas[key] = delta.clone()

    return deltas


def apply_lora_delta(
    model: torch.nn.Module,
    deltas: Dict[str, torch.Tensor]
) -> None:
    """
    Apply aggregated LoRA deltas to a model in-place.

    Args:
        model: The model to apply deltas to
        deltas: Dictionary of parameter names to delta tensors

    Example:
        >>> apply_lora_delta(base_model, aggregated_deltas)
        >>> # base_model is now updated with the aggregated deltas
    """
    with torch.no_grad():
        state_dict = model.state_dict()

        for key, delta in deltas.items():
            if key in state_dict:
                # Apply the delta to the parameter
                state_dict[key].add_(delta)


class TinyLoRAModel(torch.nn.Module):
    """Tiny model with LoRA-like parameters for testing."""

    def __init__(self, hidden_dim: int = 128, lora_rank: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.lora_A_1 = torch.nn.Parameter(torch.zeros(lora_rank, hidden_dim))
        self.lora_B_1 = torch.nn.Parameter(torch.zeros(hidden_dim, lora_rank))
        self.lora_A_2 = torch.nn.Parameter(torch.zeros(lora_rank, hidden_dim))
        self.lora_B_2 = torch.nn.Parameter(torch.zeros(hidden_dim, lora_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.T
        lora1 = x @ self.lora_A_1.T @ self.lora_B_1.T
        lora2 = x @ self.lora_A_2.T @ self.lora_B_2.T
        return base + lora1 + lora2


def create_tiny_lora_model(hidden_dim: int = 128, lora_rank: int = 4) -> TinyLoRAModel:
    """Create a tiny model with LoRA parameters for testing."""
    return TinyLoRAModel(hidden_dim=hidden_dim, lora_rank=lora_rank)


class LoRAExtractor:
    """
    Thread-safe LoRA delta extraction with state management.

    This class provides a thread-safe wrapper for LoRA delta operations,
    useful for server-side processing where multiple clients may be
    compressing their updates simultaneously.
    """

    def __init__(self):
        """Initialize the LoRA extractor with a lock for thread safety."""
        self._lock = threading.Lock()

    def extract(self, model_before: torch.nn.Module, model_after: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Thread-safe extraction of LoRA deltas.

        Args:
            model_before: The model state before training
            model_after: The model state after training

        Returns:
            Dictionary of LoRA parameter names to delta tensors
        """
        with self._lock:
            return extract_lora_delta(model_before, model_after)

    def apply(self, model: torch.nn.Module, deltas: Dict[str, torch.Tensor]) -> None:
        """
        Thread-safe application of LoRA deltas.

        Args:
            model: The model to apply deltas to
            deltas: Dictionary of parameter names to delta tensors
        """
        with self._lock:
            apply_lora_delta(model, deltas)
