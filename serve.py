#!/usr/bin/env python3
"""
serve.py — Start the Flower FL server for LeWM federated training.

Usage:
    python3 serve.py                          # default: 0.0.0.0:8080, 3 rounds, min 2 clients
    python3 serve.py --port 9090 --rounds 50  # custom
    python3 serve.py --min-clients 1          # single-device testing

Clients join with:
    python3 join.py --server <this-machine-ip>:8080
"""
import argparse
import logging

import torch
import numpy as np
from flwr.server import start_server, ServerConfig as FlowerServerConfig
from flwr.common import ndarrays_to_parameters

from server.diloco_strategy import DiLoCoStrategy
from models.lewm_config import LeWMConfig, get_simulation_config, get_ubs1_config
from models.lewm_loader import load_lewm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oyster-server")


def main():
    parser = argparse.ArgumentParser(description="Oyster FL Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--mode", default="simulation", choices=["simulation", "production"])
    args = parser.parse_args()

    cfg = get_ubs1_config() if args.mode == "production" else get_simulation_config()

    logger.info(f"Building LeWM model ({cfg.encoder.backbone}, ~{cfg.estimated_params_m:.1f}M params)")
    model = load_lewm_model(cfg)
    init_params = [p.detach().cpu().numpy() for p in model.parameters()]
    n_tensors = len(init_params)
    n_params = sum(p.numel() for p in model.parameters())
    del model

    logger.info(f"Initial parameters: {n_tensors} tensors, {n_params/1e6:.1f}M total")

    strategy = DiLoCoStrategy(
        fraction_fit=1.0,  # use all available clients
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=ndarrays_to_parameters(init_params),
        local_steps=cfg.training.local_steps,
        outer_lr=cfg.federation.outer_lr,
        outer_momentum=cfg.federation.outer_momentum,
    )

    addr = f"0.0.0.0:{args.port}"
    logger.info(f"Starting server on {addr}")
    logger.info(f"  Rounds: {args.rounds}")
    logger.info(f"  Min clients: {args.min_clients}")
    logger.info(f"  Waiting for clients to connect...")
    logger.info(f"  Clients run: python3 join.py --server <this-ip>:{args.port}")

    start_server(
        server_address=addr,
        config=FlowerServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
