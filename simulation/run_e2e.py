"""End-to-end FL simulation runner with Flower server and simulated clients.

This script runs a complete federated learning simulation:
- Flower server with DiLoCo strategy
- 10 simulated phone clients (tiny LoRA model)
- Compression pipeline (LoRA delta -> TopK 1% -> SignSGD -> msgpack)
- Client dropout simulation (randomly skip 2/10 clients per round)
"""

import argparse
import logging
import threading
import time
from typing import List, Dict
import socket

import flwr as fl
import numpy as np
import torch

from server.diloco_strategy import DiLoCoStrategy
from compressor.lora_extractor import create_tiny_lora_model
from .fl_client import create_fl_client

logger = logging.getLogger(__name__)


def get_free_port():
    """Find a free port for the Flower server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_flower_server(
    port: int,
    num_rounds: int,
    num_clients: int,
    local_steps: int,
    config: Dict
) -> None:
    """Start Flower server with DiLoCo strategy.

    Args:
        port: Port to listen on
        num_rounds: Number of FL rounds
        num_clients: Total number of clients
        local_steps: Local training steps per client
        config: Server configuration
    """
    # Create initial parameters from tiny model
    model = create_tiny_lora_model(
        hidden_dim=config.get("hidden_dim", 256),
        lora_rank=config.get("lora_rank", 4)
    )
    initial_params = [param.detach().cpu().numpy() for param in model.parameters()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_params)

    # Create DiLoCo strategy
    strategy = DiLoCoStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, num_clients - 2),  # Account for dropout
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        accept_failures=True,
        local_steps=local_steps,
        outer_lr=config.get("outer_lr", 0.7),
        outer_momentum=config.get("outer_momentum", 0.9),
        round_timeout=config.get("round_timeout", 60),
    )

    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    logger.info(f"Starting Flower server on port {port}...")
    logger.info(f"  - Number of rounds: {num_rounds}")
    logger.info(f"  - Total clients: {num_clients}")
    logger.info(f"  - Local steps: {local_steps}")
    logger.info(f"  - Outer LR: {config.get('outer_lr', 0.7)}")
    logger.info(f"  - Outer momentum: {config.get('outer_momentum', 0.9)}")

    # Start server
    fl.server.start_server(
        server_address=f"127.0.0.1:{port}",
        config=server_config,
        strategy=strategy,
    )

    logger.info("Flower server completed")


def start_flower_client(
    client_id: int,
    port: int,
    config: Dict,
    dropout_prob: float
) -> None:
    """Start a Flower client.

    Args:
        client_id: Client identifier
        port: Server port
        config: Client configuration
        dropout_prob: Probability of dropping out each round
    """
    logger.info(f"Starting client {client_id}...")

    # Create and start client
    client = create_fl_client(
        client_id=client_id,
        config=config,
        dropout_prob=dropout_prob
    )

    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=client,
    )

    logger.info(f"Client {client_id} completed")


def run_e2e_simulation(
    num_clients: int = 10,
    num_rounds: int = 3,
    local_steps: int = 5,
    hidden_dim: int = 256,
    lora_rank: int = 4,
    k_ratio: float = 0.01,
    dropout_prob: float = 0.2,
    port: int = None
) -> None:
    """Run end-to-end FL simulation.

    Args:
        num_clients: Number of simulated clients
        num_rounds: Number of FL rounds
        local_steps: Local training steps per client
        hidden_dim: Hidden dimension for tiny model
        lora_rank: LoRA rank
        k_ratio: Top-K sparsification ratio (0.01 = 1%)
        dropout_prob: Probability of client dropout per round
        port: Server port (auto-assigned if None)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get or assign port
    if port is None:
        port = get_free_port()

    # Configuration
    config = {
        "hidden_dim": hidden_dim,
        "lora_rank": lora_rank,
        "k_ratio": k_ratio,
        "outer_lr": 0.7,
        "outer_momentum": 0.9,
        "round_timeout": 60,
    }

    logger.info("=" * 70)
    logger.info("End-to-End FL Simulation")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  - Number of clients: {num_clients}")
    logger.info(f"  - Number of rounds: {num_rounds}")
    logger.info(f"  - Local steps: {local_steps}")
    logger.info(f"  - Hidden dimension: {hidden_dim}")
    logger.info(f"  - LoRA rank: {lora_rank}")
    logger.info(f"  - Top-K ratio: {k_ratio}")
    logger.info(f"  - Dropout probability: {dropout_prob}")
    logger.info(f"  - Server port: {port}")
    logger.info("=" * 70)

    # Start server in a separate thread
    server_thread = threading.Thread(
        target=start_flower_server,
        args=(port, num_rounds, num_clients, local_steps, config),
        daemon=True
    )
    server_thread.start()

    # Give server time to start
    time.sleep(2)

    # Start clients in separate threads
    client_threads = []
    for client_id in range(num_clients):
        client_thread = threading.Thread(
            target=start_flower_client,
            args=(client_id, port, config, dropout_prob),
            daemon=True
        )
        client_thread.start()
        client_threads.append(client_thread)

        # Stagger client starts slightly
        time.sleep(0.1)

    logger.info(f"All {num_clients} clients started")

    # Wait for all clients to complete
    for client_thread in client_threads:
        client_thread.join(timeout=300)

    # Wait for server to complete
    server_thread.join(timeout=300)

    logger.info("=" * 70)
    logger.info("Simulation completed successfully!")
    logger.info("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end FL simulation with Flower server and clients"
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of simulated clients (default: 10)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of FL rounds (default: 3)"
    )
    parser.add_argument(
        "--local-steps",
        type=int,
        default=5,
        help="Local training steps per client (default: 5)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for tiny model (default: 256)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=4,
        help="LoRA rank (default: 4)"
    )
    parser.add_argument(
        "--k-ratio",
        type=float,
        default=0.01,
        help="Top-K sparsification ratio (default: 0.01 for 1%)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Client dropout probability (default: 0.2 for 2/10 clients)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (auto-assigned if not specified)"
    )

    args = parser.parse_args()

    run_e2e_simulation(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_steps=args.local_steps,
        hidden_dim=args.hidden_dim,
        lora_rank=args.lora_rank,
        k_ratio=args.k_ratio,
        dropout_prob=args.dropout,
        port=args.port
    )


if __name__ == "__main__":
    main()
