"""
Qwen2.5 Model Loader with INT4 Quantization and LoRA

This module provides functionality to load Qwen2.5-1.5B-Instruct models
with INT4 quantization and LoRA (Low-Rank Adaptation) adapters using
the peft library.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict

from .quantization import get_int4_config, get_int8_config, get_fp16_config


def load_qwen25_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    quantization: str = "INT4",
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen2.5 model with specified quantization.

    Args:
        model_name: HuggingFace model identifier
        quantization: Quantization type ("INT4", "INT8", "FP16", or None for no quantization)
        device_map: Device mapping strategy ("auto", "cuda", "cpu", etc.)
        trust_remote_code: Whether to trust remote code (needed for Qwen)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_qwen25_model(quantization="INT4")
        >>> model.eval()
    """
    # Get quantization config
    if quantization == "INT4":
        quantization_config = get_int4_config()
    elif quantization == "INT8":
        quantization_config = get_int8_config()
    elif quantization == "FP16":
        quantization_config = get_fp16_config()
        # For FP16, we don't use bitsandbytes quantization config
        quantization_config = None
    else:
        quantization_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # Load model
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if quantization == "FP16":
        model_kwargs["torch_dtype"] = torch.float16
    elif quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model, tokenizer


def add_lora_to_model(
    model: AutoModelForCausalLM,
    rank: int = 4,
    alpha: int = 8,
    target_modules: Optional[list[str]] = None,
    dropout: float = 0.05,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> AutoModelForCausalLM:
    """
    Add LoRA adapters to a Qwen2.5 model.

    Args:
        model: The base model to add LoRA adapters to
        rank: LoRA rank (dimension of the low-rank adaptation)
        alpha: LoRA alpha scaling factor
        target_modules: List of module names to apply LoRA to
        dropout: Dropout probability for LoRA layers
        task_type: Task type for LoRA (CAUSAL_LM for language models)

    Returns:
        The model with LoRA adapters added

    Example:
        >>> model, tokenizer = load_qwen25_model()
        >>> model = add_lora_to_model(model, rank=4, alpha=8)
        >>> # Now model has trainable LoRA adapters
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, lora_config)
    return model


def load_qwen25_with_lora(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    quantization: str = "INT4",
    lora_rank: int = 4,
    lora_alpha: int = 8,
    lora_target_modules: Optional[list[str]] = None,
    lora_dropout: float = 0.05,
    device_map: Optional[str] = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load Qwen2.5 with LoRA adapters in one call.

    Args:
        model_name: HuggingFace model identifier
        quantization: Quantization type ("INT4", "INT8", "FP16", or None)
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        device_map: Device mapping strategy

    Returns:
        Tuple of (model with LoRA, tokenizer)

    Example:
        >>> model, tokenizer = load_qwen25_with_lora(
        ...     quantization="INT4",
        ...     lora_rank=4,
        ...     lora_alpha=8
        ... )
        >>> model.print_trainable_parameters()
    """
    model, tokenizer = load_qwen25_model(
        model_name=model_name,
        quantization=quantization,
        device_map=device_map
    )

    model = add_lora_to_model(
        model=model,
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=lora_target_modules,
        dropout=lora_dropout
    )

    return model, tokenizer


def save_model_state_dict(model: torch.nn.Module, path: str) -> Dict[str, torch.Tensor]:
    """
    Save model state dictionary for later delta extraction.

    This is used to save the base model state before training,
    which can then be compared to the trained model to extract LoRA deltas.

    Args:
        model: The model to save state from
        path: Path to save the state dictionary

    Returns:
        The state dictionary

    Example:
        >>> base_state = save_model_state_dict(base_model, "base_state.pt")
        >>> # After training...
        >>> deltas = extract_lora_delta(base_state, model.state_dict())
    """
    state_dict = model.state_dict()
    torch.save(state_dict, path)
    return state_dict


def load_model_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a previously saved model state dictionary.

    Args:
        path: Path to the saved state dictionary

    Returns:
        The loaded state dictionary

    Example:
        >>> base_state = load_model_state_dict("base_state.pt")
    """
    return torch.load(path, weights_only=True)


def get_lora_target_modules_for_qwen(
    num_layers: Optional[int] = None,
    include_mlp: bool = False
) -> list[str]:
    """
    Get the list of target modules for applying LoRA to Qwen models.

    Qwen2.5 uses a standard transformer architecture with:
    - Attention projections: q_proj, k_proj, v_proj, o_proj
    - MLP/FFN projections: gate_proj, up_proj, down_proj (optional)

    Args:
        num_layers: Number of layers to apply LoRA to (None for all layers)
        include_mlp: Whether to also apply LoRA to MLP/FFN projections

    Returns:
        List of target module names

    Example:
        >>> targets = get_lora_target_modules_for_qwen(include_mlp=False)
        >>> # Returns: ["q_proj", "k_proj", "v_proj", "o_proj"]
    """
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if include_mlp:
        attention_modules.extend(["gate_proj", "up_proj", "down_proj"])

    return attention_modules


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (trainable_params, total_params)

    Example:
        >>> trainable, total = count_trainable_parameters(model)
        >>> print(f"Trainable: {trainable:,} / {total:,}")
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


__all__ = [
    "load_qwen25_model",
    "add_lora_to_model",
    "load_qwen25_with_lora",
    "save_model_state_dict",
    "load_model_state_dict",
    "get_lora_target_modules_for_qwen",
    "count_trainable_parameters",
]
