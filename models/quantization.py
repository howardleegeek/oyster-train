"""
INT4 Quantization Utilities

This module provides utilities for INT4 quantization using bitsandbytes,
optimized for Qwen2.5 models to reduce memory footprint while maintaining
reasonable model quality.
"""

import torch
from typing import Optional


def get_int4_config(
    compute_dtype: torch.dtype = torch.float16,
    use_double_quant: bool = True,
    quant_type: str = "nf4"
) -> dict:
    """
    Get INT4 quantization configuration for bitsandbytes.

    Args:
        compute_dtype: Data type for computation (float16 is recommended)
        use_double_quant: Whether to use double quantization (quantize quantization constants)
        quant_type: Quantization type ('nf4' or 'fp4'), 'nf4' typically works better

    Returns:
        Dictionary with bitsandbytes quantization configuration

    Example:
        >>> config = get_int4_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B-Instruct",
        ...     quantization_config=config
        ... )
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_use_double_quant": use_double_quant,
        "bnb_4bit_quant_type": quant_type,
    }


def get_int8_config() -> dict:
    """
    Get INT8 quantization configuration for bitsandbytes.

    INT8 provides less aggressive compression than INT4 but typically
    maintains better model quality.

    Returns:
        Dictionary with bitsandbytes quantization configuration

    Example:
        >>> config = get_int8_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B-Instruct",
        ...     quantization_config=config
        ... )
    """
    return {
        "load_in_8bit": True,
    }


def get_fp16_config() -> dict:
    """
    Get FP16 (half precision) configuration.

    FP16 doesn't use quantization but reduces memory by storing
    parameters in 16-bit floating point format.

    Returns:
        Dictionary with torch dtype configuration

    Example:
        >>> config = get_fp16_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B-Instruct",
        ...     torch_dtype=config["torch_dtype"]
        ... )
    """
    return {
        "torch_dtype": torch.float16,
    }


def is_quantized_model(model: torch.nn.Module) -> bool:
    """
    Check if a model is quantized using bitsandbytes.

    Args:
        model: The model to check

    Returns:
        True if the model uses quantization, False otherwise
    """
    # Check if any parameter is a 4-bit or 8-bit quantized parameter
    for param in model.parameters():
        if hasattr(param, 'quant_state') or (
            hasattr(param, 'data') and
            hasattr(param.data, 'quant_state')
        ):
            return True
    return False


def get_model_memory_usage(model: torch.nn.Module) -> dict:
    """
    Get memory usage statistics for a model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with memory statistics:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - param_memory_mb: Parameter memory in MB
        - buffer_memory_mb: Buffer memory in MB
        - total_memory_mb: Total memory in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_memory / (1024 ** 2),
        "buffer_memory_mb": buffer_memory / (1024 ** 2),
        "total_memory_mb": (param_memory + buffer_memory) / (1024 ** 2),
    }


def print_memory_summary(model: torch.nn.Module, name: str = "Model") -> None:
    """
    Print a formatted memory summary for a model.

    Args:
        model: The model to summarize
        name: Name of the model for display
    """
    stats = get_model_memory_usage(model)

    print(f"\n{'=' * 50}")
    print(f"{name} Memory Summary")
    print(f"{'=' * 50}")
    print(f"Total Parameters:     {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"Parameter Memory:     {stats['param_memory_mb']:.2f} MB")
    print(f"Buffer Memory:        {stats['buffer_memory_mb']:.2f} MB")
    print(f"Total Memory:         {stats['total_memory_mb']:.2f} MB")
    print(f"{'=' * 50}\n")


def get_quantization_memory_savings(
    original_dtype: torch.dtype = torch.float32,
    target_dtype: str = "int4"
) -> float:
    """
    Calculate theoretical memory savings from quantization.

    Args:
        original_dtype: Original data type of the model
        target_dtype: Target quantization type ('int4', 'int8', 'fp16')

    Returns:
        Compression ratio (original_size / quantized_size)

    Example:
        >>> ratio = get_quantization_memory_savings(torch.float32, "int4")
        >>> print(f"Memory reduced by {ratio:.2f}x")
    """
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8,
    }

    quantization_sizes = {
        "int4": 0.5,  # 4 bits = 0.5 bytes
        "int8": 1.0,  # 8 bits = 1 byte
        "fp16": 2.0,  # 16 bits = 2 bytes
    }

    original_size = dtype_sizes.get(original_dtype, 4)
    quantized_size = quantization_sizes.get(target_dtype, 1.0)

    return original_size / quantized_size


__all__ = [
    "get_int4_config",
    "get_int8_config",
    "get_fp16_config",
    "is_quantized_model",
    "get_model_memory_usage",
    "print_memory_summary",
    "get_quantization_memory_savings",
]
