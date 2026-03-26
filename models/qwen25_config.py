"""
Qwen2.5 training configuration with Pydantic models.

Provides factory methods for different deployment scenarios:
- get_ubs1_config(): Optimal for UBS1 phone (6GB RAM)
- get_simulation_config(): CPU-friendly for testing
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quantization: Optional[Literal["INT4", "INT8", "FP16"]] = "INT4"
    max_seq_len: int = 256


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""
    rank: int = 4
    alpha: int = 8
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout: float = 0.05


class TrainingConfig(BaseModel):
    """Local training configuration."""
    lr: float = 5e-5
    batch_size: int = 4
    local_steps: int = 500
    warmup_steps: int = 50


class FederationConfig(BaseModel):
    """Federated learning configuration."""
    num_rounds: int = 100
    fraction_fit: float = 0.1
    min_clients: int = 10
    outer_lr: float = 0.7
    outer_momentum: float = 0.9


class CompressionConfig(BaseModel):
    """Gradient compression configuration."""
    k_ratio: float = 0.01
    use_signsgd: bool = True
    error_feedback: bool = True


class Qwen25Config(BaseModel):
    """Complete Qwen2.5 training configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    @classmethod
    def get_ubs1_config(cls) -> "Qwen25Config":
        """
        Returns optimal configuration for UBS1 phone (6GB RAM).

        Optimized for memory constraints while maintaining training quality.
        """
        return cls(
            model=ModelConfig(
                name="Qwen/Qwen2.5-1.5B-Instruct",
                quantization="INT4",
                max_seq_len=256
            ),
            lora=LoRAConfig(
                rank=4,
                alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                dropout=0.05
            ),
            training=TrainingConfig(
                lr=5e-5,
                batch_size=4,
                local_steps=500,
                warmup_steps=50
            ),
            federation=FederationConfig(
                num_rounds=100,
                fraction_fit=0.1,
                min_clients=10,
                outer_lr=0.7,
                outer_momentum=0.9
            ),
            compression=CompressionConfig(
                k_ratio=0.01,
                use_signsgd=True,
                error_feedback=True
            )
        )

    @classmethod
    def get_simulation_config(cls) -> "Qwen25Config":
        """
        Returns CPU-friendly configuration for testing and simulation.

        Uses smaller parameters and faster convergence for quick iteration.
        """
        return cls(
            model=ModelConfig(
                name="Qwen/Qwen2.5-0.5B-Instruct",
                quantization=None,  # FP16 on CPU for simplicity
                max_seq_len=128
            ),
            lora=LoRAConfig(
                rank=4,
                alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                dropout=0.05
            ),
            training=TrainingConfig(
                lr=1e-4,
                batch_size=2,
                local_steps=10,  # Much smaller for fast testing
                warmup_steps=5
            ),
            federation=FederationConfig(
                num_rounds=10,
                fraction_fit=0.5,
                min_clients=3,
                outer_lr=0.5,
                outer_momentum=0.9
            ),
            compression=CompressionConfig(
                k_ratio=0.05,
                use_signsgd=True,
                error_feedback=True
            )
        )

    @classmethod
    def get_tiny_config(cls) -> "Qwen25Config":
        """
        Returns configuration using tiny random model for fast integration tests.

        This creates a minimal model for quick testing without downloading
        large models from HuggingFace.
        """
        return cls(
            model=ModelConfig(
                name="tiny-random",
                quantization=None,
                max_seq_len=64
            ),
            lora=LoRAConfig(
                rank=4,
                alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                dropout=0.05
            ),
            training=TrainingConfig(
                lr=1e-3,
                batch_size=2,
                local_steps=5,
                warmup_steps=2
            ),
            federation=FederationConfig(
                num_rounds=3,
                fraction_fit=0.5,
                min_clients=3,
                outer_lr=0.5,
                outer_momentum=0.9
            ),
            compression=CompressionConfig(
                k_ratio=0.1,
                use_signsgd=True,
                error_feedback=True
            )
        )


def get_ubs1_config() -> Qwen25Config:
    """Factory method: optimal config for UBS1 phone (6GB RAM)."""
    return Qwen25Config.get_ubs1_config()


def get_simulation_config() -> Qwen25Config:
    """Factory method: CPU-friendly config for testing."""
    return Qwen25Config.get_simulation_config()


def get_tiny_config() -> Qwen25Config:
    """Factory method: tiny random model for fast integration tests."""
    return Qwen25Config.get_tiny_config()
