"""
Server-side training data collection for initial fine-tuning.
Handles downloading and mixing datasets for federated learning preparation.
"""

from typing import List, Tuple, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np


def download_alpaca_cleaned() -> Dataset:
    """
    Download the Alpaca cleaned dataset for instruction tuning.
    
    Returns:
        Dataset containing Alpaca instruction-response pairs
    """
    try:
        # Try to load from HuggingFace
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        return dataset
    except Exception as e:
        print(f"Failed to load Alpaca cleaned dataset: {e}")
        # Return a minimal dataset as fallback
        return Dataset.from_dict({
            "instruction": ["Tell me about AI."],
            "input": [""],
            "output": ["AI is a field of computer science."]
        })


def download_wikitext() -> Dataset:
    """
    Download the WikiText dataset for language modeling.
    
    Returns:
        Dataset containing Wikipedia text articles
    """
    try:
        # Load WikiText-2 dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return dataset
    except Exception as e:
        print(f"Failed to load WikiText dataset: {e}")
        # Return a minimal dataset as fallback
        return Dataset.from_dict({
            "text": ["This is a sample Wikipedia article about machine learning."]
        })


def download_chinese_alpaca() -> Dataset:
    """
    Download the Chinese Alpaca dataset for Chinese instruction tuning.
    
    Returns:
        Dataset containing Chinese instruction-response pairs
    """
    try:
        # Try to load a Chinese instruction dataset
        dataset = load_dataset("shibing624/alpaca-chinese", split="train")
        return dataset
    except Exception as e:
        print(f"Failed to load Chinese Alpaca dataset: {e}")
        # Return a minimal dataset as fallback
        return Dataset.from_dict({
            "instruction": ["请介绍一下人工智能。"],
            "input": [""],
            "output": ["人工智能是计算机科学的一个分支。"]
        })


def create_mixed_dataset(
    datasets: List[Dataset], 
    weights: List[float],
    split_ratio: float = 0.9
) -> DatasetDict:
    """
    Mix datasets with specified weights and split into train/validation.
    
    Args:
        datasets: List of datasets to mix
        weights: List of weights for each dataset (should sum to 1.0)
        split_ratio: Ratio of data to use for training (rest for validation)
        
    Returns:
        DatasetDict containing 'train' and 'validation' splits
    """
    if len(datasets) != len(weights):
        raise ValueError("Number of datasets must match number of weights")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        print(f"Normalized weights to: {weights}")
    
    # Calculate number of samples to take from each dataset
    total_samples = max(len(d) for d in datasets)  # Use largest dataset as reference
    samples_per_dataset = [int(total_samples * w) for w in weights]
    
    # Adjust for rounding errors
    diff = total_samples - sum(samples_per_dataset)
    if diff != 0:
        samples_per_dataset[0] += diff  # Add remainder to first dataset
    
    # Sample from each dataset
    sampled_datasets = []
    for dataset, num_samples in zip(datasets, samples_per_dataset):
        if len(dataset) > num_samples:
            # Random sampling
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            sampled = dataset.select(indices)
        else:
            # Use all samples if dataset is smaller than requested
            sampled = dataset
        sampled_datasets.append(sampled)
    
    # Concatenate all sampled datasets
    from datasets import concatenate_datasets
    mixed_dataset = concatenate_datasets(sampled_datasets)
    
    # Shuffle the mixed dataset
    mixed_dataset = mixed_dataset.shuffle(seed=42)
    
    # Split into train and validation
    split_index = int(len(mixed_dataset) * split_ratio)
    train_dataset = mixed_dataset.select(range(split_index))
    validation_dataset = mixed_dataset.select(range(split_index, len(mixed_dataset)))
    
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })


# On-device data specification
ON_DEVICE_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source": {"type": "string"},
        "timestamp": {"type": "integer"},
        "language": {"type": "string", "enum": ["zh", "en", "mixed"]}
    },
    "required": ["text", "source", "timestamp", "language"]
}

ON_DEVICE_DATA_DESCRIPTION = """
On-device data specification for federated learning clients:

Each data point should conform to the following schema:
{
    "text": str,          # The actual text content
    "source": str,        # Source of the data (e.g., "sms", "notes", "browser")
    "timestamp": int,     # Unix timestamp when data was created/captured
    "language": str       # Language code: "zh" (Chinese), "en" (English), or "mixed"
}

Example:
{
    "text": "今天天气不错，适合去公园散步。",
    "source": "notes",
    "timestamp": 1640995200,
    "language": "zh"
}
"""


if __name__ == "__main__":
    # Test dataset downloading and mixing
    print("Testing dataset functions...")
    
    # Test individual download functions (will use fallbacks if datasets not available)
    alpaca = download_alpaca_cleaned()
    print(f"Alpaca dataset size: {len(alpaca)}")
    print(f"Alpaca example: {alpaca[0] if len(alpaca) > 0 else 'Empty'}")
    
    wikitext = download_wikitext()
    print(f"WikiText dataset size: {len(wikitext)}")
    print(f"WikiText example: {wikitext[0] if len(wikitext) > 0 else 'Empty'}")
    
    chinese_alpaca = download_chinese_alpaca()
    print(f"Chinese Alpaca dataset size: {len(chinese_alpaca)}")
    print(f"Chinese Alpaca example: {chinese_alpaca[0] if len(chinese_alpaca) > 0 else 'Empty'}")
    
    # Test mixed dataset creation
    try:
        mixed = create_mixed_dataset(
            [alpaca, wikitext, chinese_alpaca],
            [0.5, 0.3, 0.2]  # 50% Alpaca, 30% WikiText, 20% Chinese Alpaca
        )
        print(f"\nMixed dataset:")
        print(f"Train size: {len(mixed['train'])}")
        print(f"Validation size: {len(mixed['validation'])}")
        print(f"Train example: {mixed['train'][0] if len(mixed['train']) > 0 else 'Empty'}")
    except Exception as e:
        print(f"Error creating mixed dataset: {e}")
    
    # Print on-device data specification
    print("\n" + "="*50)
    print("ON-DEVICE DATA SPECIFICATION")
    print("="*50)
    print(ON_DEVICE_DATA_DESCRIPTION)