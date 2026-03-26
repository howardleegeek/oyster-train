"""
Flower client for LeWM federated world model training.

Each phone client:
1. Receives global model parameters from server
2. Trains LeWM on local camera/sensor data for N steps
3. Computes parameter delta (no LoRA — full params, model is only 15M)
4. Compresses delta via Top-K + SignSGD pipeline
5. Sends compressed delta back to server
"""
import gc
import logging
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from compressor import CompressionPipeline
from models.lewm_config import LeWMConfig, get_ubs1_config, get_simulation_config
from models.lewm_loader import load_lewm_model, extract_delta

logger = logging.getLogger(__name__)


class SyntheticWorldDataset:
    """Generates synthetic pixel+action sequences for simulation.

    In production, this is replaced by the Android camera pipeline.
    """

    def __init__(self, num_samples: int, cfg: LeWMConfig):
        self.num_samples = num_samples
        self.seq_len = cfg.data.sequence_length
        self.img_size = cfg.encoder.image_size
        self.action_dim = cfg.data.action_dim * cfg.data.frameskip

    def get_dataloader(self, batch_size: int) -> DataLoader:
        pixels = torch.randn(self.num_samples, self.seq_len, 3, self.img_size, self.img_size)
        actions = torch.randn(self.num_samples, self.seq_len, self.action_dim)
        dataset = TensorDataset(pixels, actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class LeWMPhoneClient(fl.client.NumPyClient):
    """Flower client for on-device LeWM training."""

    def __init__(
        self,
        client_id: int,
        cfg: LeWMConfig,
        device: str = "cpu",
        num_samples: int = 200,
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.device = device

        # Build model
        self.model = load_lewm_model(cfg).to(device)

        # Data (synthetic for sim, camera feed on real phone)
        self.dataset = SyntheticWorldDataset(num_samples, cfg)

        # Compression — no LoRA extraction needed, just raw deltas
        self.compression = CompressionPipeline(k_ratio=cfg.federation.compression_k_ratio)

        # Snapshot for delta computation
        self._checkpoint = None

        logger.info(f"LeWM client {client_id} ready on {device}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters as numpy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load global parameters from server."""
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype, device=self.device)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training round.

        1. Load global params
        2. Snapshot params (for delta)
        3. Train locally for local_steps
        4. Compute + compress delta
        5. Return compressed delta
        """
        # Load global model
        self.set_parameters(parameters)

        # Snapshot before training
        before = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

        # Local training
        local_steps = config.get("local_steps", self.cfg.training.local_steps)
        batch_size = self.cfg.training.batch_size
        lr = self.cfg.training.lr

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=self.cfg.training.weight_decay
        )
        dataloader = self.dataset.get_dataloader(batch_size)

        total_loss = 0.0
        steps = 0
        for epoch in range(max(1, local_steps // len(dataloader) + 1)):
            for pixels, actions in dataloader:
                if steps >= local_steps:
                    break
                pixels = pixels.to(self.device)
                actions = actions.to(self.device)

                output = self.model(pixels, actions)
                loss = output["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"Client {self.client_id}: {steps} steps, loss={avg_loss:.4f}")

        # Compute delta
        after = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        delta = extract_delta(before, after)

        # Return delta as parameter list
        delta_arrays = [delta[k].numpy() for k in sorted(delta.keys())]

        # Clean up
        del before, after, delta
        gc.collect()

        metrics = {
            "loss": avg_loss,
            "steps": steps,
            "client_id": self.client_id,
        }

        return delta_arrays, self.dataset.num_samples, metrics

    def run_evaluation(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Run model assessment on local data."""
        self.set_parameters(parameters)
        self.model.eval()

        dataloader = self.dataset.get_dataloader(self.cfg.training.batch_size)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for pixels, actions in dataloader:
                pixels = pixels.to(self.device)
                actions = actions.to(self.device)
                output = self.model(pixels, actions)
                total_loss += output["loss"].item() * pixels.size(0)
                total_samples += pixels.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, total_samples, {"loss": avg_loss}
