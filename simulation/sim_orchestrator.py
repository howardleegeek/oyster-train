"""Orchestrator for federated learning simulation.

Coordinates the Flower server and multiple simulated phone clients.
"""

import os
import argparse
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from flwr.client import ClientProxy
    from flwr.common import FitRes, EvaluateRes, Parameters, Scalar
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import yaml

from .sim_client import PhoneClient
from .data_loader import create_client_datasets, PhoneDataset
from compressor import CompressionPipeline

logger = logging.getLogger(__name__)


class SimServer:
    """Flower server for simulation."""

    def __init__(self, config: Dict):
        """Initialize server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.num_rounds = config.get("num_rounds", 10)
        self.results_dir = Path("simulation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.round_metrics = []

        # Global LoRA parameters
        self.global_params = None
        self.compression = CompressionPipeline(
            k_ratio=config.get("k_ratio", 0.01)
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple["ClientProxy", "FitRes"]],
        failures: List[BaseException]
    ) -> Tuple[Optional["Parameters"], Dict[str, "Scalar"]]:
        """Aggregate client updates using FedAvg.

        Args:
            server_round: Current round number
            results: List of (client, FitRes) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            logger.warning(f"Round {server_round}: No client results received")
            return None, {}

        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")

        # Extract parameters and metrics
        client_params = []
        client_examples = []
        round_metrics = {
            "round": server_round,
            "num_clients": len(results),
            "clients": []
        }

        for client_proxy, fit_res in results:
            # Get parameters from client
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_params.append(params)

            # Get metrics
            num_examples = fit_res.num_examples
            client_examples.append(num_examples)

            client_metrics = fit_res.metrics
            round_metrics["clients"].append(client_metrics)

            logger.debug(
                f"Client {client_metrics.get('client_id', 'unknown')}: "
                f"loss={client_metrics.get('loss', 0):.4f}, "
                f"mem={client_metrics.get('peak_memory_mb', 0):.1f}MB"
            )

        # Aggregate using weighted average (FedAvg)
        aggregated_params = self._weighted_fedavg(client_params, client_examples)

        # Store as global parameters
        self.global_params = aggregated_params

        # Compute round statistics
        total_examples = sum(client_examples)
        avg_loss = np.mean([m["loss"] for m in round_metrics["clients"]])
        avg_memory = np.mean([m["peak_memory_mb"] for m in round_metrics["clients"]])
        avg_compression = np.mean([m["compression_ratio"] for m in round_metrics["clients"]])
        total_bytes_saved = sum([
            m.get("original_size_bytes", 0) - m.get("compressed_size_bytes", 0)
            for m in round_metrics["clients"]
        ])

        round_metrics.update({
            "total_examples": total_examples,
            "avg_loss": avg_loss,
            "avg_peak_memory_mb": avg_memory,
            "avg_compression_ratio": avg_compression,
            "total_bytes_saved": total_bytes_saved
        })

        self.round_metrics.append(round_metrics)

        logger.info(
            f"Round {server_round}: Avg Loss={avg_loss:.4f}, "
            f"Avg Memory={avg_memory:.1f}MB, "
            f"Avg Compression={avg_compression:.2f}x, "
            f"Saved {total_bytes_saved / 1024:.1f}KB"
        )

        # Convert to Flower parameters
        fl_params = fl.common.ndarrays_to_parameters(aggregated_params)

        return fl_params, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple["ClientProxy", "EvaluateRes"]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, "Scalar"]]:
        """Aggregate evaluation metrics.

        Args:
            server_round: Current round number
            results: List of (client, EvaluateRes) tuples
            failures: List of failures

        Returns:
            Tuple of (loss, metrics)
        """
        if not results:
            return None, {}

        # Compute weighted average
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        weighted_loss = sum(
            evaluate_res.num_examples * evaluate_res.loss
            for _, evaluate_res in results
        ) / total_examples

        # Compute average accuracy
        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        avg_accuracy = np.mean(accuracies)

        logger.info(
            f"Round {server_round} Evaluation: "
            f"Loss={weighted_loss:.4f}, Accuracy={avg_accuracy:.4f}"
        )

        return weighted_loss, {
            "accuracy": avg_accuracy,
            "num_clients": len(results)
        }

    def _weighted_fedavg(
        self,
        client_params: List[List[np.ndarray]],
        client_examples: List[int]
    ) -> List[np.ndarray]:
        """Weighted FedAvg aggregation.

        Args:
            client_params: List of client parameter arrays
            client_examples: List of sample counts per client

        Returns:
            Aggregated parameters
        """
        # Initialize aggregated parameters
        aggregated = [np.zeros_like(p) for p in client_params[0]]

        # Weighted sum
        total_examples = sum(client_examples)
        for params, num_examples in zip(client_params, client_examples):
            weight = num_examples / total_examples
            for i, param in enumerate(params):
                aggregated[i] += weight * param

        return aggregated

    def save_results(self) -> None:
        """Save training results to disk."""
        results_file = self.results_dir / "training_metrics.json"
        with open(results_file, "w") as f:
            json.dump(self.round_metrics, f, indent=2)

        # Save summary
        summary = {
            "num_rounds": self.num_rounds,
            "total_clients": self.config.get("num_clients", 100),
            "final_loss": self.round_metrics[-1]["avg_loss"] if self.round_metrics else 0,
            "compression_ratio": self.compression.get_compression_ratio()
        }

        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {self.results_dir}")


def client_fn(
    client_id: str,
    train_datasets: List[PhoneDataset],
    val_datasets: List[PhoneDataset],
    config: Dict
) -> "fl.client.Client":
    """Create a Flower client.

    Args:
        client_id: Client identifier
        train_datasets: List of training datasets
        val_datasets: List of validation datasets
        config: Client configuration

    Returns:
        Flower client instance
    """
    idx = int(client_id)
    client = PhoneClient(
        client_id=idx,
        train_dataset=train_datasets[idx],
        val_dataset=val_datasets[idx],
        config=config
    )
    return client.to_client()


def run_simulation(
    num_clients: int = 100,
    num_rounds: int = 10,
    config_path: str = "simulation/configs/default.yaml"
) -> None:
    """Run federated learning simulation.

    Args:
        num_clients: Number of simulated phone clients
        num_rounds: Number of federated training rounds
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    config["num_clients"] = num_clients
    config["num_rounds"] = num_rounds

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info(f"Starting simulation with {num_clients} clients for {num_rounds} rounds")
    logger.info(f"Configuration: {config}")

    # Create non-IID datasets
    logger.info("Creating non-IID data shards...")
    train_datasets, val_datasets = create_client_datasets(
        num_clients=num_clients,
        alpha=config.get("data_alpha", 0.5),
        tokenizer_name=config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_seq_len=config.get("max_seq_len", 256),
        dataset_name=config.get("dataset", "wikitext")
    )

    # Create server
    server = SimServer(config)

    # Initialize global parameters (all zeros)
    global_params = [np.zeros(1, dtype=np.float32)]

    # Start simulation using flwr simulation
    start_time = time.time()

    logger.info("Starting Flower simulation...")

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(
            cid, train_datasets, val_datasets, config
        ),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=int(num_clients * 0.1),  # Sample 10% of clients
            min_evaluate_clients=int(num_clients * 0.1),
            min_available_clients=int(num_clients * 0.1),
            fit_metrics_aggregation_fn=lambda metrics: {
                k: sum(v for v in values) / len(values)
                for k, values in zip(metrics.keys(), metrics.values())
            },
            evaluate_metrics_aggregation_fn=lambda metrics: {
                k: sum(v for v in values) / len(values)
                for k, values in zip(metrics.keys(), metrics.values())
            }
        ),
        client_resources={"num_cpus": 2},
    )

    wall_time = time.time() - start_time

    # Save results
    server.save_results()

    # Log summary
    logger.info(f"Simulation completed in {wall_time:.1f}s")
    logger.info(f"Results saved to simulation/results/")

    # Print training curve
    if server.round_metrics:
        print("\nTraining Summary:")
        print("-" * 60)
        print(f"{'Round':<6} {'Clients':<8} {'Avg Loss':<12} {'Avg Memory (MB)':<15}")
        print("-" * 60)
        for metric in server.round_metrics:
            print(
                f"{metric['round']:<6} "
                f"{metric['num_clients']:<8} "
                f"{metric['avg_loss']:<12.4f} "
                f"{metric['avg_peak_memory_mb']:<15.1f}"
            )
        print("-" * 60)

        compression_ratio = server.compression.get_compression_ratio()
        print(f"\nCompression Statistics:")
        print(f"  Compression ratio: {compression_ratio:.2f}x")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run federated learning simulation for phone clients"
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=100,
        help="Number of simulated phone clients (default: 100)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated training rounds (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="simulation/configs/default.yaml",
        help="Path to configuration file (default: simulation/configs/default.yaml)"
    )

    args = parser.parse_args()

    run_simulation(
        num_clients=args.clients,
        num_rounds=args.rounds,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
