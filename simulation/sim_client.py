"""Flower client for federated learning simulation.

Implements a phone client that performs local LoRA fine-tuning on Qwen2.5-0.5B.
"""

import gc
import os
from typing import Dict, Tuple, List
import logging

import flwr as fl
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from compressor import CompressionPipeline
from .data_loader import PhoneDataset

logger = logging.getLogger(__name__)


class PhoneClient(fl.client.NumPyClient):
    """Flower client simulating a phone performing LoRA fine-tuning."""

    def __init__(
        self,
        client_id: int,
        train_dataset: PhoneDataset,
        val_dataset: PhoneDataset,
        config: Dict,
        device: str = "cpu"
    ):
        """Initialize phone client.

        Args:
            client_id: Unique client identifier
            train_dataset: Local training data shard
            val_dataset: Local validation data shard
            config: Client configuration
            device: Device to run on (cpu/cuda)
        """
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        # Model and LoRA components
        self.model = None
        self.tokenizer = None
        self.lora_config = LoraConfig(
            r=config.get("lora_rank", 4),
            lora_alpha=config.get("lora_alpha", 8),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Compression
        self.compression = CompressionPipeline(
            topk_ratio=config.get("topk_ratio", 0.01)
        )

        # Memory tracking
        self.peak_memory = 0

        logger.info(f"Client {client_id} initialized with {len(train_dataset)} train samples")

    def get_parameters(self, config: Dict) -> list[np.ndarray]:
        """Return current LoRA parameters.

        Args:
            config: Server configuration

        Returns:
            List of LoRA parameter arrays
        """
        if self.model is None:
            logger.warning(f"Client {self.client_id}: Model not initialized, returning empty params")
            return []

        # Extract only LoRA parameters
        params = []
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                params.append(param.detach().cpu().numpy())

        return params

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set LoRA parameters from server.

        Args:
            parameters: List of parameter arrays from server
        """
        if self.model is None:
            return

        # Set LoRA parameters
        param_idx = 0
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param_idx < len(parameters):
                param.data = torch.tensor(
                    parameters[param_idx],
                    dtype=param.dtype,
                    device=param.device
                )
                param_idx += 1

    def fit(self, parameters: list[np.ndarray], config: Dict) -> Tuple[list, int, Dict]:
        """Train locally on client data shard.

        Args:
            parameters: Global model parameters from server
            config: Training configuration including local_steps

        Returns:
            Tuple of (compressed_parameters, num_examples, metrics)
        """
        local_steps = config.get("local_steps", self.config.get("local_steps", 500))
        batch_size = self.config.get("batch_size", 4)

        logger.info(
            f"Client {self.client_id}: Starting training for {local_steps} steps, "
            f"batch_size={batch_size}"
        )

        # Load model
        self._load_model()
        self.set_parameters(parameters)

        # Store initial parameters for delta computation
        initial_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if "lora" in name.lower()
        }

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Setup training
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01
        )

        # Training loop
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        step = 0
        while step < local_steps:
            for batch in train_loader:
                if step >= local_steps:
                    break

                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                step += 1

                # Log memory every 50 steps
                if step % 50 == 0:
                    self._log_memory()

        # Log final memory
        self._log_memory()

        # Compute delta (difference from initial params)
        delta_params = []
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and name in initial_params:
                delta = (param.detach().cpu() - initial_params[name]).numpy()
                delta_params.append(delta)

        # Compress delta
        compressed_delta, compression_meta = self.compression.compress(delta_params)

        # Clean up
        del initial_params, delta_params
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / num_batches if num_batches > 0 else 0,
            "num_examples": len(self.train_dataset),
            "peak_memory_mb": self.peak_memory,
            "compression_ratio": compression_meta.get("compression_ratio", 1.0),
            "bytes_saved": compression_meta.get("bytes_saved", 0)
        }

        logger.info(
            f"Client {self.client_id}: Training complete. "
            f"Loss={metrics['loss']:.4f}, "
            f"Peak Memory={metrics['peak_memory_mb']:.1f}MB, "
            f"Compression={metrics['compression_ratio']:.2f}x"
        )

        return compressed_delta, len(self.train_dataset), metrics

    def evaluate(self, parameters: list[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate on local validation shard.

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting evaluation")

        # Load model
        self._load_model()
        self.set_parameters(parameters)

        # Create data loader
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=False
        )

        # Evaluation loop
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy (next token prediction)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # Only count non-padded tokens
                labels = batch["labels"]
                mask = labels != -100

                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0

        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_examples": len(self.val_dataset)
        }

        logger.info(
            f"Client {self.client_id}: Evaluation complete. "
            f"Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}"
        )

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, len(self.val_dataset), metrics

    def _load_model(self) -> None:
        """Load Qwen2.5-0.5B model with LoRA."""
        if self.model is not None:
            return

        model_name = self.config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")

        logger.info(f"Client {self.client_id}: Loading {model_name}")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        # Add LoRA
        self.model = get_peft_model(self.model, self.lora_config)

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        logger.info(f"Client {self.client_id}: Model loaded with LoRA")

    def _log_memory(self) -> None:
        """Log current memory usage."""
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / 1024 / 1024
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.peak_memory = max(self.peak_memory, peak_mb)
            logger.debug(
                f"Client {self.client_id}: "
                f"Memory: {current_mb:.1f}MB (peak: {peak_mb:.1f}MB)"
            )
        else:
            # Estimate memory for CPU
            import psutil
            process = psutil.Process(os.getpid())
            current_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_mb)
            logger.debug(f"Client {self.client_id}: Memory: {current_mb:.1f}MB")
