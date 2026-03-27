"""
End-to-end integration test for Oyster Phone Training Protocol.

Tests the complete pipeline: compression -> aggregation using tiny models.
Does NOT require HuggingFace model downloads or GPU.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock
from server.diloco_strategy import DiLoCoStrategy
from compressor.pipeline import CompressionPipeline
from compressor.lora_extractor import (
    extract_lora_delta, apply_lora_delta, create_tiny_lora_model
)
from flwr.common import (
    Parameters, FitRes, ndarrays_to_parameters,
    parameters_to_ndarrays, Status, Code
)


@pytest.fixture
def tiny_global_model():
    """Create tiny global model for testing."""
    return create_tiny_lora_model()


def _simulate_local_training(model, steps=10, lr=1e-3):
    """Simulate local training by perturbing LoRA params."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(torch.randn_like(param) * lr)


def test_compression_roundtrip(tiny_global_model):
    """Test: extract -> compress -> decompress preserves structure."""
    model_before = create_tiny_lora_model()
    model_after = create_tiny_lora_model()

    _simulate_local_training(model_after)

    deltas = extract_lora_delta(model_before, model_after)
    assert len(deltas) > 0

    pipeline = CompressionPipeline(k_ratio=0.01)
    compressed = pipeline.compress(deltas)
    decompressed = pipeline.decompress(compressed)

    for name in deltas:
        assert decompressed[name].shape == deltas[name].shape

    ratio = pipeline.get_compression_ratio()
    assert ratio > 1.0


def test_diloco_aggregation_with_compression():
    """Test: multiple client updates -> DiLoCo aggregation."""
    strategy = DiLoCoStrategy(
        outer_lr=0.5,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=3,
        min_available_clients=5,
        fraction_fit=0.6,
    )

    results = []
    for i in range(5):
        params = [np.random.randn(10, 10).astype(np.float32)]
        mock_client = Mock()
        mock_client.cid = f"client_{i}"
        fit_result = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
            num_examples=50,
            metrics={"loss": 0.5 - i * 0.05}
        )
        results.append((mock_client, fit_result))

    losses = []
    for round_num in range(1, 4):
        aggregated_params, metrics = strategy.aggregate_fit(
            round_num, results, []
        )
        assert aggregated_params is not None
        assert metrics["num_clients"] == 5
        losses.append(metrics.get("loss", 0))

    assert len(losses) == 3
    assert strategy.velocity is not None


def test_end_to_end_tiny_model():
    """
    Full end-to-end test with tiny model:
    1. Create global model
    2. Simulate 3 local trainings
    3. Extract + compress deltas
    4. Aggregate with DiLoCo
    5. Verify global model changes
    """
    global_model = create_tiny_lora_model()

    strategy = DiLoCoStrategy(
        outer_lr=0.5,
        outer_momentum=0.9,
        local_steps=10,
        min_fit_clients=2,
    )

    pipeline = CompressionPipeline(k_ratio=0.05)

    client_updates = []
    for i in range(3):
        local_model = create_tiny_lora_model()
        local_model.load_state_dict(global_model.state_dict())
        _simulate_local_training(local_model, steps=10, lr=1e-2)

        deltas = extract_lora_delta(global_model, local_model)
        compressed_bytes = pipeline.compress(deltas)
        decompressed = pipeline.decompress(compressed_bytes)

        param_arrays = [v.numpy() for v in decompressed.values()]
        client_updates.append(param_arrays)

    results = []
    for i, params in enumerate(client_updates):
        mock_client = Mock()
        mock_client.cid = f"phone_{i}"
        fit_result = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
            num_examples=100,
            metrics={"loss": 0.5}
        )
        results.append((mock_client, fit_result))

    aggregated_params, metrics = strategy.aggregate_fit(1, results, [])
    assert aggregated_params is not None

    final_ndarrays = parameters_to_ndarrays(aggregated_params)
    has_nonzero = any(np.abs(arr).sum() > 0 for arr in final_ndarrays)
    assert has_nonzero, "Aggregated parameters should be non-zero"


def test_compression_pipeline_thread_safety():
    """Test that compression pipeline is thread-safe."""
    from threading import Thread

    pipeline = CompressionPipeline(k_ratio=0.01)
    errors = []

    def compress_worker(worker_id):
        try:
            deltas = {f"layer_{worker_id}": torch.randn(1000)}
            compressed = pipeline.compress(deltas)
            decompressed = pipeline.decompress(compressed)
            assert f"layer_{worker_id}" in decompressed
        except Exception as e:
            errors.append(e)

    threads = [Thread(target=compress_worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"


def test_compression_ratio_target():
    """Test: 3MB LoRA delta compresses to substantial ratio."""
    deltas = {}
    for layer in range(28):
        for proj in ["q", "k", "v", "o"]:
            deltas[f"layers.{layer}.{proj}_proj.lora_A"] = torch.randn(4, 1536)
            deltas[f"layers.{layer}.{proj}_proj.lora_B"] = torch.randn(1536, 4)

    original_bytes = sum(t.numel() * 4 for t in deltas.values())

    pipeline = CompressionPipeline(k_ratio=0.01)
    compressed = pipeline.compress(deltas)
    ratio = pipeline.get_compression_ratio()

    assert ratio > 50, f"Ratio {ratio:.1f}x too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
