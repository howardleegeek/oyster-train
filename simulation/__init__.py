"""Simulation module for federated learning."""

from .sim_client import PhoneClient
from .data_loader import PhoneDataset, create_non_iid_shards, create_client_datasets
from .sim_orchestrator import run_simulation

__all__ = [
    "PhoneClient",
    "PhoneDataset",
    "create_non_iid_shards",
    "create_client_datasets",
    "run_simulation"
]
