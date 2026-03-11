"""DiLoCo strategy for Flower federated learning server."""
import time
import logging
from typing import Optional, Tuple, Dict, List, Union
from collections import OrderedDict

import numpy as np
import torch
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    EvaluateRes,
    Config,
    NDArray,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager

from server.config import config

logger = logging.getLogger(__name__)


class DiLoCoStrategy(FedAvg):
    """DiLoCo (Distributed Low-Communication) strategy for federated learning.

    This strategy implements the DiLoCo protocol where clients perform multiple
    local steps (default 500) before syncing with the server. The server aggregates
    compressed LoRA deltas using an outer optimizer with Nesterov momentum.

    Features:
    - Outer optimizer with Nesterov momentum (beta=0.9, lr=0.7)
    - Straggler handling with configurable timeout
    - Round tracking and logging
    - Compressed delta aggregation
    """

    def __init__(
        self,
        *,
        fraction_fit: float = config.fraction_fit,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = config.min_clients,
        min_evaluate_clients: int = 0,
        min_available_clients: int = config.min_available_clients,
        on_fit_config_fn: Optional[Config] = None,
        on_evaluate_config_fn: Optional[Config] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        local_steps: int = config.local_steps,
        outer_lr: float = config.outer_lr,
        outer_momentum: float = config.outer_momentum,
        round_timeout: int = config.round_timeout,
    ) -> None:
        """Initialize DiLoCo strategy.

        Args:
            fraction_fit: Fraction of clients used during training.
            fraction_evaluate: Fraction of clients used during evaluation.
            min_fit_clients: Minimum number of clients used during training.
            min_evaluate_clients: Minimum number of clients used during evaluation.
            min_available_clients: Minimum number of total clients in the system.
            on_fit_config_fn: Function used to configure training.
            on_evaluate_config_fn: Function used to configure evaluation.
            accept_failures: Whether or not accept failures during training.
            initial_parameters: Initial global model parameters.
            fit_metrics_aggregation_fn: Aggregation function for fit metrics.
            evaluate_metrics_aggregation_fn: Aggregation function for evaluate metrics.
            local_steps: Number of local steps clients run before syncing.
            outer_lr: Learning rate for the outer optimizer.
            outer_momentum: Momentum coefficient for the outer optimizer.
            round_timeout: Timeout in seconds for each round.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.local_steps = local_steps
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.round_timeout = round_timeout
        self.current_round = 0
        self.velocity: Optional[List[NDArray]] = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> Tuple[List[Tuple[ClientProxy, Config]], Config]:
        """Configure the next round of training.

        Args:
            server_round: The current round of federated learning.
            parameters: The current model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A tuple of clients and config to be used for training in this round.
        """
        self.current_round = server_round

        # Use parent's configure_fit to get client configuration
        clients_and_config = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Configure each client with local steps and timeout
        config_dict = {
            "local_steps": self.local_steps,
            "timeout": self.round_timeout,
            "current_round": server_round,
        }

        # Update the config for each client
        updated_clients = [
            (client, {**cfg, **config_dict}) for client, cfg in clients_and_config
        ]

        logger.info(
            f"Round {server_round}: Configured {len(updated_clients)} clients "
            f"with {self.local_steps} local steps"
        )

        return updated_clients, config_dict

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters using DiLoCo outer optimizer.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the selected clients.
            failures: Failures from the selected clients.

        Returns:
            Tuple containing the aggregated parameters and metrics.
        """
        start_time = time.time()

        # Log participants
        participating_clients = [client.cid for client, _ in results]
        logger.info(
            f"Round {server_round}: {len(participating_clients)} clients participated, "
            f"{len(failures)} failures"
        )

        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {"aggregation_time": 0.0, "loss": float("inf")}

        # Extract parameters and metrics from results
        parameters_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        losses = [fit_res.metrics.get("loss", float("inf")) for _, fit_res in results]

        # Use parent's aggregate_fit to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            logger.warning(f"Round {server_round}: No aggregated parameters returned")
            return None, aggregated_metrics

        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # Apply DiLoCo outer optimizer with Nesterov momentum
        if self.velocity is None:
            # Initialize velocity on first round
            self.velocity = [np.zeros_like(arr) for arr in aggregated_ndarrays]

        # Apply Nesterov momentum update
        for i, (params, vel) in enumerate(zip(aggregated_ndarrays, self.velocity)):
            # Pseudo-gradient from aggregated update
            pseudo_gradient = params

            # Nesterov momentum update: v_t = beta * v_{t-1} + (1 - beta) * grad
            # Apply to global model with learning rate
            self.velocity[i] = (
                self.outer_momentum * vel + (1 - self.outer_momentum) * pseudo_gradient
            )

            # Apply update with learning rate
            aggregated_ndarrays[i] = self.velocity[i] * self.outer_lr

        # Convert back to Parameters
        updated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        aggregation_time = time.time() - start_time
        avg_loss = np.mean([l for l in losses if l != float("inf")])

        metrics = {
            "aggregation_time": aggregation_time,
            "loss": float(avg_loss),
            "num_clients": len(results),
            "num_failures": len(failures),
            "participating_clients": participating_clients,
        }

        logger.info(
            f"Round {server_round}: Aggregation completed in {aggregation_time:.2f}s, "
            f"loss: {avg_loss:.4f}"
        )

        return updated_parameters, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Config]]:
        """Configure evaluation on held-out validation set.

        Args:
            server_round: The current round of federated learning.
            parameters: The current model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A list of tuples containing clients and config to be used for evaluation.
        """
        # Configure evaluation
        config_dict = {
            "eval_round": server_round,
        }

        # Use parent's configure_evaluate to get client configuration
        clients_and_config = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        # Update config for each client
        updated_clients = [(client, {**cfg, **config_dict}) for client, cfg in clients_and_config]

        logger.info(
            f"Round {server_round}: Configured {len(updated_clients)} clients for evaluation"
        )

        return updated_clients

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average.

        Args:
            server_round: The current round of federated learning.
            results: Successful evaluations from the selected clients.
            failures: Failures from the selected clients.

        Returns:
            Tuple containing the aggregated loss and metrics.
        """
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        logger.info(
            f"Round {server_round}: Evaluation loss: {aggregated_loss:.4f}, "
            f"{len(results)} clients"
        )

        return aggregated_loss, aggregated_metrics
