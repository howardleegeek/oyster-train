"""Configuration settings for the Flower server."""
from typing import Optional
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Server configuration using Pydantic settings.

    All settings can be overridden via environment variables.
    """

    flower_port: int = 8080
    min_clients: int = 10
    fraction_fit: float = 0.1
    local_steps: int = 500
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    round_timeout: int = 300
    total_rounds: int = 100
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank: int = 4
    lora_alpha: int = 8
    min_available_clients: int = 50

    class Config:
        env_prefix = "FLOWER_"
        env_file = ".env"
        extra = "ignore"


# Global config instance
config = ServerConfig()


def get_server_config(**overrides) -> ServerConfig:
    """Factory to create a ServerConfig with optional overrides."""
    return ServerConfig(**overrides)
