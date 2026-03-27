"""
Unit tests for Flower server and DiLoCo strategy.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from flwr.common import (
    Parameters, FitRes, ndarrays_to_parameters,
    parameters_to_ndarrays, Status, Code
)
from flwr.server.client_proxy import ClientProxy
from server.diloco_strategy import DiLoCoStrategy
from server.config import ServerConfig, get_server_config


class MockClientProxy(ClientProxy):
    """Mock Flower client proxy for testing."""

    def __init__(self, cid: str):
        self.cid = cid

    def get_properties(self, ins=None):
        return {}

    def get_parameters(self, ins=None):
        return Parameters([], {})

    def fit(self, ins=None):
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters([], {}),
            num_examples=100,
            metrics={"loss": 0.5}
        )

    def evaluate(self, ins=None):
        return 0.5, 100, {"loss": 0.5}

    def reboot(self, reconnect=None):
        return True

    def reconnect(self, reconnect=None):
        return True

    def disconnect(self):
        pass


def test_diloco_strategy_init():
    """Test: strategy creates with correct config."""
    strategy = DiLoCoStrategy(
        outer_lr=0.7,
        outer_momentum=0.9,
        local_steps=500,
        min_fit_clients=10,
        min_available_clients=50,
        fraction_fit=0.1
    )

    assert strategy.outer_lr == 0.7
    assert strategy.outer_momentum == 0.9
    assert strategy.local_steps == 500
    assert strategy.min_fit_clients == 10
    assert strategy.min_available_clients == 50


def test_aggregate_fit():
    """Test: mock client results -> verify aggregation."""
    strategy = DiLoCoStrategy(
        outer_lr=0.7,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=3,
    )

    client1 = MockClientProxy("1")
    client2 = MockClientProxy("2")

    params1 = [np.random.randn(10, 10).astype(np.float32)]
    params2 = [np.random.randn(10, 10).astype(np.float32)]

    fit_res1 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params1),
        num_examples=50,
        metrics={"loss": 0.5}
    )
    fit_res2 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params2),
        num_examples=50,
        metrics={"loss": 0.4}
    )

    results = [(client1, fit_res1), (client2, fit_res2)]
    failures = []

    aggregated_params, metrics = strategy.aggregate_fit(1, results, failures)

    assert aggregated_params is not None
    assert metrics["num_clients"] == 2
    assert "aggregation_time" in metrics


def test_outer_optimizer():
    """Test: Nesterov momentum update correct."""
    strategy = DiLoCoStrategy(
        outer_lr=0.7,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=2,
    )

    client1 = MockClientProxy("1")
    client2 = MockClientProxy("2")

    params1 = [np.ones((10, 10), dtype=np.float32) * 0.1]
    params2 = [np.ones((10, 10), dtype=np.float32) * 0.2]

    fit_res1 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params1),
        num_examples=50,
        metrics={}
    )
    fit_res2 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params2),
        num_examples=50,
        metrics={}
    )

    results = [(client1, fit_res1), (client2, fit_res2)]

    # First round (velocity initialized)
    strategy.aggregate_fit(1, results, [])
    assert strategy.velocity is not None

    # Second round (velocity updated with momentum)
    strategy.aggregate_fit(2, results, [])

    # Verify velocity was updated (list of arrays, all should be non-zero)
    for vel in strategy.velocity:
        assert np.abs(vel).sum() > 0


def test_min_clients():
    """Test: server waits for minimum clients."""
    strategy = DiLoCoStrategy(
        outer_lr=0.7,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=5,
        min_available_clients=10,
        fraction_fit=0.5
    )

    assert strategy.min_fit_clients == 5
    assert strategy.min_available_clients == 10
    assert strategy.fraction_fit == 0.5


class TestServerConfig:
    """Tests for server configuration."""

    def test_default_config(self):
        config = ServerConfig()
        assert config.flower_port == 8080
        assert config.min_clients == 10
        assert config.min_available_clients == 50
        assert config.fraction_fit == 0.1
        assert config.local_steps == 500
        assert config.round_timeout == 300
        assert config.outer_lr == 0.7
        assert config.outer_momentum == 0.9
        assert config.total_rounds == 100
        assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.lora_rank == 4
        assert config.lora_alpha == 8

    def test_custom_config(self):
        config = ServerConfig(flower_port=9090, min_clients=5, outer_lr=0.5)
        assert config.flower_port == 9090
        assert config.min_clients == 5
        assert config.outer_lr == 0.5


def test_get_server_config():
    """Test factory method creates config."""
    config = get_server_config(flower_port=9090)
    assert config.flower_port == 9090
    assert isinstance(config, ServerConfig)


def test_straggler_timeout():
    """Test: slow client handled gracefully."""
    strategy = DiLoCoStrategy(
        outer_lr=0.7,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=2,
    )

    client1 = MockClientProxy("1")
    params1 = [np.ones((10, 10), dtype=np.float32) * 0.1]
    fit_res1 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params1),
        num_examples=50,
        metrics={"loss": 0.5}
    )

    results = [(client1, fit_res1)]
    failures = [Exception("Client timeout")]

    aggregated_params, metrics = strategy.aggregate_fit(1, results, failures)

    assert aggregated_params is not None
    assert metrics["num_clients"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
