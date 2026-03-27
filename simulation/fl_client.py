"""Flower client for end-to-end FL simulation with tiny LoRA model."""

import logging
import random
from typing import Dict, Tuple, List
import copy

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from compressor import CompressionPipeline
from compressor.lora_extractor import create_tiny_lora_model, extract_lora_delta, apply_lora_delta

logger = logging.getLogger(__name__)


class TinyFLClient(fl.client.NumPyClient):
    """Flower client simulating a phone with tiny LoRA model.

    This client uses a tiny mock model (not real Qwen) for testing purposes.
    Each client performs 5 local training steps per round with simulated gradients.
    """

    def __init__(
        self,
        client_id: int,
        config: Dict,
        dropout_prob: float = 0.2
    ):
        """Initialize tiny FL client.

        Args:
            client_id: Unique client identifier
            config: Client configuration
            dropout_prob: Probability of client dropping out each round
        """
        self.client_id = client_id
        self.config = config
        self.dropout_prob = dropout_prob
        self.should_participate = True

        # Create tiny LoRA model
        hidden_dim = config.get("hidden_dim", 256)
        lora_rank = config.get("lora_rank", 4)
        self.model = create_tiny_lora_model(hidden_dim=hidden_dim, lora_rank=lora_rank)
        self.model.eval()

        # Compression pipeline
        self.compression_pipeline = CompressionPipeline(k_ratio=config.get("k_ratio", 0.01))

        # Store model state before training for delta computation
        self.model_before = None

        logger.info(f"Client {client_id}: Initialized with tiny LoRA model "
                    f"(hidden_dim={hidden_dim}, lora_rank={lora_rank})")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current LoRA parameters.

        Args:
            config: Server configuration

        Returns:
            List of LoRA parameter arrays
        """
        params = []
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                params.append(param.detach().cpu().numpy())
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set LoRA parameters from server.

        Args:
            parameters: List of parameter arrays from server
        """
        param_idx = 0
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param_idx < len(parameters):
                param.data = torch.tensor(
                    parameters[param_idx],
                    dtype=param.dtype,
                    device=param.device
                )
                param_idx += 1

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train locally on client data shard.

        Args:
            parameters: Global model parameters from server
            config: Training configuration including local_steps

        Returns:
            Tuple of (compressed_parameters, num_examples, metrics)
        """
        # Handle client dropout (skip 2/10 clients randomly)
        if random.random() < self.dropout_prob:
            self.should_participate = False
            logger.info(f"Client {self.client_id}: Dropping out this round")
            # Return zero-valued deltas (same shape as parameters) so Flower
            # can still aggregate without shape mismatch errors
            zero_deltas = [np.zeros_like(p) for p in parameters]
            return zero_deltas, 0, {
                "client_id": self.client_id,
                "dropout": True,
                "loss": 0.0,
                "compression_ratio": 0.0
            }

        self.should_participate = True
        local_steps = config.get("local_steps", 5)

        logger.info(f"Client {self.client_id}: Starting training for {local_steps} steps")

        # Set parameters from server
        self.set_parameters(parameters)

        # Store model state before training
        self.model_before = copy.deepcopy(self.model)

        # Simulate local training with random gradients
        # In real scenario, this would train on actual data
        self.model.train()
        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.01
        )

        total_loss = 0.0
        for step in range(local_steps):
            # Simulate forward/backward pass with random gradients
            # Create fake input and target
            batch_size = 8
            x = torch.randn(batch_size, self.model.weight.shape[1])
            target = torch.randn(batch_size, self.model.weight.shape[1])

            optimizer.zero_grad()

            # Forward pass
            output = self.model(x)
            loss = nn.MSELoss()(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / local_steps

        # Compute LoRA delta
        lora_deltas = extract_lora_delta(self.model_before, self.model)

        # Compute original size for compression ratio
        original_size = sum(
            param.element_size() * param.nelement()
            for name, param in self.model_before.named_parameters()
            if "lora" in name
        )

        # Compress delta
        compressed_bytes = self.compression_pipeline.compress(lora_deltas, original_size=original_size)
        compression_ratio = self.compression_pipeline.get_compression_ratio()

        # Convert compressed bytes back to numpy arrays for Flower protocol
        # Note: In real scenario, we'd send compressed bytes directly
        # For Flower, we need to send parameters. Since we're simulating,
        # we'll send the delta parameters directly (uncompressed for Flower protocol)
        delta_params = []
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                delta = (param.data - self.model_before.state_dict()[name]).detach().cpu().numpy()
                delta_params.append(delta)

        metrics = {
            "client_id": self.client_id,
            "dropout": False,
            "loss": avg_loss,
            "num_examples": 8 * local_steps,
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "compressed_size_bytes": len(compressed_bytes)
        }

        logger.info(
            f"Client {self.client_id}: Training complete. "
            f"Loss={avg_loss:.4f}, Compression={compression_ratio:.2f}x"
        )

        return delta_params, 8 * local_steps, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate on local validation shard.

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)

        # Simulate evaluation with random loss
        loss = random.uniform(0.5, 2.0)

        metrics = {
            "client_id": self.client_id,
            "loss": loss,
            "num_examples": 100
        }

        logger.info(f"Client {self.client_id}: Evaluation complete. Loss={loss:.4f}")

        return loss, 100, metrics


def create_fl_client(
    client_id: int,
    config: Dict,
    dropout_prob: float = 0.2
) -> fl.client.Client:
    """Create a Flower client.

    Args:
        client_id: Client identifier
        config: Client configuration
        dropout_prob: Probability of client dropping out

    Returns:
        Flower client instance
    """
    client = TinyFLClient(client_id=client_id, config=config, dropout_prob=dropout_prob)
    return client.to_client()
