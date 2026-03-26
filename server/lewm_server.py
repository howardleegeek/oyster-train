"""
Flower server entry point for LeWM federated world model training.

Uses DiLoCoStrategy with full-parameter aggregation (no LoRA needed
since LeWM is only ~15M params vs Qwen2.5's 1.5B).
"""
import logging
from typing import Optional

import torch
import numpy as np
from flwr.server import start_server, ServerConfig as FlowerServerConfig
from flwr.common import ndarrays_to_parameters

from server.diloco_strategy import DiLoCoStrategy
from models.lewm_config import LeWMConfig, get_simulation_config, get_ubs1_config
from models.lewm_loader import load_lewm_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_lewm_initial_parameters(cfg: LeWMConfig) -> Optional[list]:
    """Create initial LeWM parameters for the server.

    Unlike Qwen2.5 (which needs HuggingFace download), LeWM builds
    from scratch in <1 second with random init.
    """
    logger.info("Building LeWM model for initial parameters...")

    model = load_lewm_model(cfg)
    parameters = [p.detach().cpu().numpy() for p in model.parameters()]

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initial parameters: {len(parameters)} tensors, {n_params/1e6:.1f}M total")

    del model
    return parameters


def main(mode: str = "simulation") -> None:
    """Start the LeWM federated learning server.

    Args:
        mode: "simulation" for CPU testing, "production" for phone deployment
    """
    if mode == "simulation":
        cfg = get_simulation_config()
    else:
        cfg = get_ubs1_config()

    logger.info(f"Starting LeWM FL server (mode={mode})")
    logger.info(f"  Encoder: {cfg.encoder.backbone}")
    logger.info(f"  Params: ~{cfg.estimated_params_m:.1f}M")
    logger.info(f"  Memory: ~{cfg.estimated_memory_mb:.0f}MB per client")
    logger.info(f"  Rounds: {cfg.federation.num_rounds}")
    logger.info(f"  Min clients: {cfg.federation.min_clients}")

    initial_params = create_lewm_initial_parameters(cfg)
    initial_parameters = ndarrays_to_parameters(initial_params) if initial_params else None

    strategy = DiLoCoStrategy(
        fraction_fit=cfg.federation.fraction_fit,
        min_fit_clients=cfg.federation.min_clients,
        min_available_clients=cfg.federation.min_clients,
        initial_parameters=initial_parameters,
        local_steps=cfg.training.local_steps,
        outer_lr=cfg.federation.outer_lr,
        outer_momentum=cfg.federation.outer_momentum,
    )

    start_server(
        server_address="0.0.0.0:8080",
        config=FlowerServerConfig(num_rounds=cfg.federation.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    main(mode=mode)
