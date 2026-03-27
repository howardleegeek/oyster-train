#!/usr/bin/env python3
"""
End-to-end federated learning test.
Starts a Flower server + 2 simulated clients, runs 2 rounds.
"""
import threading
import time
import logging

import torch
import numpy as np
import flwr as fl
from flwr.server import start_server, ServerConfig
from flwr.common import ndarrays_to_parameters

from models.lewm_config import get_simulation_config
from models.lewm_loader import load_lewm_model, extract_delta
from server.diloco_strategy import DiLoCoStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e-test")

NUM_CLIENTS = 2
NUM_ROUNDS = 2
PORT = 18080  # non-standard to avoid conflicts


class TestClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.cfg = get_simulation_config()
        self.model = load_lewm_model(self.cfg)
        logger.info(f"Client {cid} ready")

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        # Load global params
        for p, v in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(v, dtype=p.dtype)

        before = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Train 5 steps
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        for step in range(5):
            px = torch.randn(2, 8, 3, 96, 96)
            act = torch.randn(2, 8, 6)
            out = self.model(px, act)
            out["loss"].backward()
            opt.step()
            opt.zero_grad()

        after = {k: v.clone() for k, v in self.model.state_dict().items()}
        delta = extract_delta(before, after)
        arrays = [delta[k].numpy() for k in sorted(delta.keys())]

        logger.info(f"Client {self.cid} trained: loss={out['loss'].item():.4f}")
        return arrays, 10, {"loss": out["loss"].item()}

    def evaluate(self, parameters, config):
        return 0.0, 10, {"loss": 0.0}


def run_server():
    cfg = get_simulation_config()
    model = load_lewm_model(cfg)
    init = [p.detach().cpu().numpy() for p in model.parameters()]
    del model

    strategy = DiLoCoStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=ndarrays_to_parameters(init),
        local_steps=5,
        outer_lr=0.7,
        outer_momentum=0.9,
    )

    logger.info(f"Server starting on port {PORT}")
    start_server(
        server_address=f"0.0.0.0:{PORT}",
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    logger.info("Server finished")


def run_client(cid):
    time.sleep(3)  # wait for server to start
    client = TestClient(cid)
    logger.info(f"Client {cid} connecting to localhost:{PORT}")
    fl.client.start_numpy_client(
        server_address=f"127.0.0.1:{PORT}",
        client=client,
    )
    logger.info(f"Client {cid} finished")


def main():
    logger.info(f"=== E2E Federated Test: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds ===")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    client_threads = []
    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=run_client, args=(i,))
        t.start()
        client_threads.append(t)

    for t in client_threads:
        t.join(timeout=120)

    server_thread.join(timeout=10)
    logger.info("=== E2E TEST PASSED ===")


if __name__ == "__main__":
    main()
