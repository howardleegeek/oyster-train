"""Model configuration and loading utilities.

Supports two model backends:
- Qwen2.5-1.5B: LLM fine-tuning via LoRA (original)
- LeWM (~15M): JEPA world model for physical environment prediction (new)
"""

from models.qwen25_config import Qwen25Config, get_ubs1_config, get_simulation_config, get_tiny_config
from models.qwen25_loader import (
    load_qwen25_model,
    add_lora_to_model,
    load_qwen25_with_lora,
    save_model_state_dict,
    load_model_state_dict,
    get_lora_target_modules_for_qwen,
    count_trainable_parameters,
)
from models.quantization import (
    get_int4_config,
    get_int8_config,
    get_fp16_config,
    is_quantized_model,
    get_model_memory_usage,
    print_memory_summary,
    get_quantization_memory_savings,
)

from models.lewm_config import (
    LeWMConfig,
    get_ubs1_config as get_lewm_ubs1_config,
    get_simulation_config as get_lewm_simulation_config,
    get_gpu_config as get_lewm_gpu_config,
)
from models.lewm_loader import (
    LeWM,
    load_lewm_model,
    get_model_state as get_lewm_state,
    set_model_state as set_lewm_state,
    extract_delta as extract_lewm_delta,
    count_trainable_parameters as count_lewm_parameters,
)

__all__ = [
    # Qwen2.5 Config
    "Qwen25Config",
    "get_ubs1_config",
    "get_simulation_config",
    "get_tiny_config",
    # Qwen2.5 Loader
    "load_qwen25_model",
    "add_lora_to_model",
    "load_qwen25_with_lora",
    "save_model_state_dict",
    "load_model_state_dict",
    "get_lora_target_modules_for_qwen",
    "count_trainable_parameters",
    # Quantization
    "get_int4_config",
    "get_int8_config",
    "get_fp16_config",
    "is_quantized_model",
    "get_model_memory_usage",
    "print_memory_summary",
    "get_quantization_memory_savings",
    # LeWM
    "LeWMConfig",
    "LeWM",
    "load_lewm_model",
    "get_lewm_ubs1_config",
    "get_lewm_simulation_config",
    "get_lewm_gpu_config",
    "get_lewm_state",
    "set_lewm_state",
    "extract_lewm_delta",
    "count_lewm_parameters",
]
