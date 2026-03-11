"""Flower federated learning server entry point."""
import logging
from typing import Optional

import torch
from transformers import AutoConfig
from flwr.server import start_server, ServerConfig as FlowerServerConfig
from flwr.common import ndarrays_to_parameters

from server.config import config
from server.diloco_strategy import DiLoCoStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_initial_parameters() -> Optional[list]:
    """Create initial model parameters with Qwen2.5-1.5B-Instruct skeleton.

    Returns:
        List of numpy arrays representing model parameter shapes (no actual weights).
    """
    logger.info(f"Loading model config for {config.model_name}")

    try:
        model_config = AutoConfig.from_pretrained(config.model_name)
        logger.info(f"Model config loaded: {model_config.model_type}")

        # Create a skeleton model without loading weights (for parameter shapes only)
        # We don't need actual weights for the server - only the structure
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(model_config)
        logger.info(f"Model skeleton created with {sum(p.numel() for p in model.parameters())} parameters")

        # Convert model parameters to numpy arrays (just for structure)
        # In practice, clients will upload LoRA deltas and we aggregate them
        parameters = [param.detach().cpu().numpy() for param in model.parameters()]
        logger.info(f"Created initial parameters with {len(parameters)} tensors")

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()

        return parameters

    except Exception as e:
        logger.error(f"Error creating initial parameters: {e}")
        logger.info("Proceeding without initial parameters")
        return None


def main() -> None:
    """Start the Flower federated learning server."""
    logger.info("Starting Flower server with DiLoCo strategy")
    logger.info(f"Configuration:")
    logger.info(f"  - Port: {config.flower_port}")
    logger.info(f"  - Min clients: {config.min_clients}")
    logger.info(f"  - Min available clients: {config.min_available_clients}")
    logger.info(f"  - Fraction fit: {config.fraction_fit}")
    logger.info(f"  - Local steps: {config.local_steps}")
    logger.info(f"  - Outer learning rate: {config.outer_lr}")
    logger.info(f"  - Outer momentum: {config.outer_momentum}")
    logger.info(f"  - Round timeout: {config.round_timeout}s")
    logger.info(f"  - Total rounds: {config.total_rounds}")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - LoRA rank: {config.lora_rank}")
    logger.info(f"  - LoRA alpha: {config.lora_alpha}")

    # Create initial parameters
    initial_parameters = None
    try:
        parameters_list = create_initial_parameters()
        if parameters_list is not None:
            initial_parameters = ndarrays_to_parameters(parameters_list)
            logger.info("Initial parameters set successfully")
    except Exception as e:
        logger.warning(f"Could not create initial parameters: {e}")
        logger.info("Server will wait for client parameters")

    # Create DiLoCo strategy
    strategy = DiLoCoStrategy(
        fraction_fit=config.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=config.min_clients,
        min_evaluate_clients=0,
        min_available_clients=config.min_available_clients,
        initial_parameters=initial_parameters,
        accept_failures=True,
        local_steps=config.local_steps,
        outer_lr=config.outer_lr,
        outer_momentum=config.outer_momentum,
        round_timeout=config.round_timeout,
    )

    # Configure and start the server
    flower_config = FlowerServerConfig(
        num_rounds=config.total_rounds,
        round_timeout=config.round_timeout,
    )

    logger.info(f"Starting Flower server on port {config.flower_port}...")
    logger.info(f"Waiting for at least {config.min_available_clients} clients to connect...")

    start_server(
        server_address=f"0.0.0.0:{config.flower_port}",
        config=flower_config,
        strategy=strategy,
    )

    logger.info("Flower server shutdown")


if __name__ == "__main__":
    main()
