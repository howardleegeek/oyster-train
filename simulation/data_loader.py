"""Data loader for federated learning simulation.

Creates non-IID data shards using Dirichlet distribution to simulate
realistic federated learning scenarios where each client has different data.
"""

import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class PhoneDataset(Dataset):
    """Dataset wrapper for phone clients with proper tokenization for Qwen2.5."""

    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_len: int = 256,
        split: str = "train"
    ):
        """Initialize phone dataset.

        Args:
            texts: List of text samples
            tokenizer_name: HuggingFace tokenizer name
            max_seq_len: Maximum sequence length (memory constraint)
            split: Dataset split (train/val)
        """
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.split = split

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training example.

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal language modeling, labels = input_ids
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Shift labels for causal LM (next token prediction)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_non_iid_shards(
    dataset: List[str],
    num_clients: int,
    alpha: float = 0.5
) -> List[List[str]]:
    """Create non-IID data shards using Dirichlet distribution.

    Args:
        dataset: List of text samples to shard
        num_clients: Number of clients to create shards for
        alpha: Dirichlet parameter (smaller = more heterogeneous)
            alpha=0.5 = moderately heterogeneous (realistic for phone users)

    Returns:
        List of data shards, one per client
    """
    num_samples = len(dataset)

    # Sample from Dirichlet distribution to get proportions
    proportions = np.random.dirichlet([alpha] * num_clients)

    # Calculate number of samples per client
    client_samples = (proportions * num_samples).astype(int)

    # Handle rounding errors
    remaining = num_samples - client_samples.sum()
    if remaining > 0:
        # Distribute remaining samples to clients with largest proportions
        largest_clients = np.argsort(-proportions)[:remaining]
        client_samples[largest_clients] += 1

    # Create index array and shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Create shards
    shards = []
    start_idx = 0
    for i, num_samples_client in enumerate(client_samples):
        end_idx = start_idx + num_samples_client
        client_indices = indices[start_idx:end_idx]
        shard = [dataset[idx] for idx in client_indices]
        shards.append(shard)
        start_idx = end_idx

    # Log distribution statistics
    sizes = [len(shard) for shard in shards]
    logger.info(
        f"Created {num_clients} non-IID shards (alpha={alpha}): "
        f"min={min(sizes)}, max={max(sizes)}, "
        f"mean={np.mean(sizes):.1f}, std={np.std(sizes):.1f}"
    )

    return shards


def load_wikitext_sample() -> Tuple[List[str], List[str]]:
    """Load a small sample of WikiText-2 for simulation.

    Returns:
        Tuple of (train_texts, val_texts)
    """
    try:
        from datasets import load_dataset

        # Load WikiText-2 (small dataset suitable for CPU simulation)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        # Extract text samples (skip empty ones)
        train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50]
        val_texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 50]

        # Limit size for faster simulation
        train_texts = train_texts[:5000]
        val_texts = val_texts[:500]

        logger.info(f"Loaded {len(train_texts)} train and {len(val_texts)} val samples")
        return train_texts, val_texts

    except Exception as e:
        logger.warning(f"Failed to load WikiText: {e}, using synthetic data")
        # Fallback to synthetic data
        return _create_synthetic_data()


def _create_synthetic_data() -> Tuple[List[str], List[str]]:
    """Create synthetic training data for simulation.

    Returns:
        Tuple of (train_texts, val_texts)
    """
    # Simple synthetic text data
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Federated learning enables training on distributed devices.",
        "Deep neural networks have revolutionized artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Optimization algorithms are essential for training models.",
        "Gradient descent is a fundamental optimization technique.",
        "Transformers have become the dominant architecture for NLP.",
        "LoRA enables efficient fine-tuning of large language models."
    ]

    # Expand with variations
    train_texts = []
    for i in range(1000):
        base = base_texts[i % len(base_texts)]
        variation = f"{base} This is sample {i} with some additional context to make it longer."
        train_texts.append(variation)

    # Validation data
    val_texts = [
        "The cat sat on the mat.",
        "Neural networks learn patterns from data.",
        "Distributed systems communicate over networks.",
        "Data preprocessing is a crucial step in ML pipelines."
    ] * 25

    logger.info(f"Created {len(train_texts)} synthetic train and {len(val_texts)} val samples")
    return train_texts, val_texts


def create_client_datasets(
    num_clients: int,
    alpha: float = 0.5,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_len: int = 256,
    dataset_name: str = "wikitext"
) -> Tuple[List[PhoneDataset], List[PhoneDataset]]:
    """Create non-IID datasets for all clients.

    Args:
        num_clients: Number of clients
        alpha: Dirichlet parameter for non-IID distribution
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        dataset_name: Dataset to use ('wikitext' or 'synthetic')

    Returns:
        Tuple of (train_datasets, val_datasets)
    """
    # Load data
    train_texts, val_texts = load_wikitext_sample()

    # Create non-IID shards
    train_shards = create_non_iid_shards(train_texts, num_clients, alpha)
    val_shards = create_non_iid_shards(val_texts, num_clients, alpha)

    # Create datasets
    train_datasets = [
        PhoneDataset(shard, tokenizer_name, max_seq_len, "train")
        for shard in train_shards
    ]
    val_datasets = [
        PhoneDataset(shard, tokenizer_name, max_seq_len, "val")
        for shard in val_shards
    ]

    return train_datasets, val_datasets
