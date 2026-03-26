"""Tests for end-to-end FL simulation.

This test validates:
- All rounds complete without error
- Compression ratio > 100x
- Aggregated model weights change each round
- Metrics logged per round (loss, num_clients, compression_ratio)
"""

import logging
import socket
import threading
import time
from typing import Dict, List, Tuple

import pytest
import torch
import numpy as np
import flwr as fl

from server.diloco_strategy import DiLoCoStrategy
from compressor.lora_extractor import create_tiny_lora_model
from simulation.fl_client import TinyFLClient

logger = logging.getLogger(__name__)


class E2ESimulationTest:
    """Test class for end-to-end FL simulation."""

    def __init__(self):
        self.rounds_completed = 0
        self.round_metrics = []
        self.global_params_history = []
        self.lock = threading.Lock()
        self.test_results = {
            "all_rounds_completed": False,
            "compression_ratio_gt_100": False,
            "weights_changed_each_round": False,
            "metrics_logged": False
        }

    def record_round(self, round_num: int, metrics: Dict, params: List[np.ndarray]):
        """Record round metrics and parameters."""
        with self.lock:
            self.rounds_completed += 1
            self.round_metrics.append({
                "round": round_num,
                "metrics": metrics.copy()
            })
            self.global_params_history.append([p.copy() for p in params])
            logger.info(f"Round {round_num} completed: {metrics}")


class DiLoCoStrategyWithRecording(DiLoCoStrategy):
    """DiLoCo strategy that records metrics for testing."""

    def __init__(self, test_instance: E2ESimulationTest, **kwargs):
        super().__init__(**kwargs)
        self.test_instance = test_instance

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List
    ) -> Tuple[fl.common.Parameters, Dict]:
        """Aggregate and record metrics."""
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Convert to numpy for comparison
        params = fl.common.parameters_to_ndarrays(aggregated_params)

        # Record the round
        self.test_instance.record_round(server_round, metrics, params)

        return aggregated_params, metrics


def get_free_port() -> int:
    """Find a free port for the Flower server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def run_test_server(
    port: int,
    num_rounds: int,
    test_instance: E2ESimulationTest
) -> None:
    """Run test server with DiLoCo strategy."""
    # Create initial parameters
    model = create_tiny_lora_model(hidden_dim=256, lora_rank=4)
    initial_params = [param.detach().cpu().numpy() for param in model.parameters()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_params)

    # Create strategy with test recording
    strategy = DiLoCoStrategyWithRecording(
        test_instance=test_instance,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=7,  # Account for dropout (10 - 2 - 1 buffer)
        min_evaluate_clients=0,
        min_available_clients=10,
        initial_parameters=initial_parameters,
        accept_failures=True,
        local_steps=5,
        outer_lr=0.7,
        outer_momentum=0.9,
        round_timeout=60,
    )

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    try:
        fl.server.start_server(
            server_address=f"127.0.0.1:{port}",
            config=server_config,
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"Server error: {e}")


def run_test_client(client_id: int, port: int, dropout_prob: float) -> None:
    """Run a test client."""
    config = {
        "hidden_dim": 256,
        "lora_rank": 4,
        "k_ratio": 0.01,
    }

    client = TinyFLClient(
        client_id=client_id,
        config=config,
        dropout_prob=dropout_prob
    )

    try:
        fl.client.start_client(
            server_address=f"127.0.0.1:{port}",
            client=client.to_client(),
        )
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")


def test_e2e_simulation():
    """Test end-to-end FL simulation.

    Validates:
    1. All rounds complete without error
    2. Compression ratio > 100x
    3. Aggregated model weights change each round
    4. Metrics logged per round (loss, num_clients, compression_ratio)
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize test instance
    test_instance = E2ESimulationTest()

    # Configuration
    num_rounds = 3
    num_clients = 10
    dropout_prob = 0.2  # Randomly skip 2/10 clients per round

    # Get free port
    port = get_free_port()

    logger.info(f"Starting test with port {port}, {num_rounds} rounds, {num_clients} clients")

    # Start server thread
    server_thread = threading.Thread(
        target=run_test_server,
        args=(port, num_rounds, test_instance),
        daemon=True
    )
    server_thread.start()

    # Give server time to start
    time.sleep(3)

    # Start client threads
    client_threads = []
    for client_id in range(num_clients):
        client_thread = threading.Thread(
            target=run_test_client,
            args=(client_id, port, dropout_prob),
            daemon=True
        )
        client_thread.start()
        client_threads.append(client_thread)
        time.sleep(0.1)

    # Wait for clients to complete (with timeout)
    for client_thread in client_threads:
        client_thread.join(timeout=180)

    # Wait for server to complete (with timeout)
    server_thread.join(timeout=180)

    logger.info(f"Rounds completed: {test_instance.rounds_completed}")
    logger.info(f"Round metrics: {test_instance.round_metrics}")

    # Test 1: All rounds completed without error
    assert test_instance.rounds_completed == num_rounds, \
        f"Expected {num_rounds} rounds, got {test_instance.rounds_completed}"
    test_instance.test_results["all_rounds_completed"] = True
    logger.info("✓ Test 1 passed: All rounds completed without error")

    # Test 2: Compression ratio > 100x
    compression_ratios = [
        metric["metrics"].get("compression_ratio", 0)
        for metric in test_instance.round_metrics
    ]
    avg_compression_ratio = np.mean([r for r in compression_ratios if r > 0])

    assert avg_compression_ratio > 100, \
        f"Expected compression ratio > 100x, got {avg_compression_ratio:.2f}x"
    test_instance.test_results["compression_ratio_gt_100"] = True
    logger.info(f"✓ Test 2 passed: Compression ratio > 100x (avg: {avg_compression_ratio:.2f}x)")

    # Test 3: Aggregated model weights change each round
    if len(test_instance.global_params_history) >= 2:
        weights_changed = True
        for i in range(1, len(test_instance.global_params_history)):
            params_prev = test_instance.global_params_history[i-1]
            params_curr = test_instance.global_params_history[i]

            # Check if any parameter changed
            changed = False
            for p_prev, p_curr in zip(params_prev, params_curr):
                if not np.allclose(p_prev, p_curr, atol=1e-6):
                    changed = True
                    break

            if not changed:
                weights_changed = False
                break

        assert weights_changed, "Model weights did not change between rounds"
        test_instance.test_results["weights_changed_each_round"] = True
        logger.info("✓ Test 3 passed: Model weights changed each round")
    else:
        test_instance.test_results["weights_changed_each_round"] = False
        logger.warning("⚠ Test 3 skipped: Not enough rounds to compare weights")

    # Test 4: Metrics logged per round
    metrics_logged = True
    for metric in test_instance.round_metrics:
        round_metrics = metric["metrics"]
        # Check for required metrics
        if not all(key in round_metrics for key in ["loss", "num_clients"]):
            metrics_logged = False
            break

        # Check if compression ratio is logged (may be 0 for dropout rounds)
        if "compression_ratio" not in round_metrics:
            metrics_logged = False
            break

    assert metrics_logged, "Required metrics not logged per round"
    test_instance.test_results["metrics_logged"] = True
    logger.info("✓ Test 4 passed: Metrics logged per round")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("E2E Simulation Test Results:")
    logger.info("=" * 70)
    for test_name, passed in test_instance.test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    logger.info("=" * 70)


def test_tiny_model_creation():
    """Test that tiny LoRA model can be created."""
    model = create_tiny_lora_model(hidden_dim=256, lora_rank=4)

    # Check model structure
    assert hasattr(model, 'weight'), "Model missing weight parameter"
    assert hasattr(model, 'lora_A_1'), "Model missing lora_A_1"
    assert hasattr(model, 'lora_B_1'), "Model missing lora_B_1"
    assert hasattr(model, 'lora_A_2'), "Model missing lora_A_2"
    assert hasattr(model, 'lora_B_2'), "Model missing lora_B_2"

    # Check shapes
    assert model.weight.shape == (256, 256), f"Unexpected weight shape: {model.weight.shape}"
    assert model.lora_A_1.shape == (4, 256), f"Unexpected lora_A_1 shape: {model.lora_A_1.shape}"
    assert model.lora_B_1.shape == (256, 4), f"Unexpected lora_B_1 shape: {model.lora_B_1.shape}"

    # Test forward pass
    x = torch.randn(8, 256)
    output = model(x)
    assert output.shape == (8, 256), f"Unexpected output shape: {output.shape}"

    logger.info("✓ Tiny model creation test passed")


def test_client_basic_operations():
    """Test basic client operations."""
    config = {
        "hidden_dim": 256,
        "lora_rank": 4,
        "k_ratio": 0.01,
    }

    client = TinyFLClient(client_id=0, config=config, dropout_prob=0.0)

    # Test get_parameters
    params = client.get_parameters({})
    assert len(params) > 0, "Client returned no parameters"
    assert all(isinstance(p, np.ndarray) for p in params), "Parameters not numpy arrays"

    # Test set_parameters
    client.set_parameters(params)

    # Test fit (simulate one round)
    fit_params, num_examples, metrics = client.fit(params, {"local_steps": 2})

    assert num_examples > 0, "Client returned 0 examples"
    assert "loss" in metrics, "Missing loss in metrics"
    assert "compression_ratio" in metrics, "Missing compression_ratio in metrics"

    logger.info("✓ Client basic operations test passed")


if __name__ == "__main__":
    # Run all tests
    test_tiny_model_creation()
    test_client_basic_operations()
    test_e2e_simulation()
    logger.info("\nAll tests passed!")
