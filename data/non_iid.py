"""
Tools for creating and analyzing non-IID data distributions for federated learning.
Implements Dirichlet partitioning, heterogeneity analysis, and FedProx regularizer.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math


class HeterogeneityMetrics:
    """Container for heterogeneity analysis results."""
    
    def __init__(
        self,
        emd_scores: List[float],
        label_skew: List[Dict[Any, float]],
        vocab_overlap: List[float],
        num_clients: int
    ):
        self.emd_scores = emd_scores
        self.label_skew = label_skew
        self.vocab_overlap = vocab_overlap
        self.num_clients = num_clients
    
    def __repr__(self):
        return f"HeterogeneityMetrics(clients={self.num_clients}, avg_emd={np.mean(self.emd_scores):.4f})"


class DirichletPartitioner:
    """
    Partition dataset into non-IID shards using Dirichlet distribution.
    
    alpha < 1.0 = more heterogeneous (realistic)
    alpha > 1.0 = more homogeneous (easier to train)
    """
    
    def __init__(self, num_clients: int, alpha: float = 0.5, seed: int = 42):
        """
        Initialize Dirichlet partitioner.
        
        Args:
            num_clients: Number of clients to partition data for
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        np.random.seed(seed)
    
    def partition(
        self, 
        dataset: Dataset, 
        labels: Optional[List[Any]] = None
    ) -> List[Subset]:
        """
        Partition dataset into non-IID shards.
        
        Args:
            dataset: Dataset to partition
            labels: List of labels for each data point (if None, uses uniform distribution)
            
        Returns:
            List of Subset objects, one per client
        """
        if labels is None:
            # If no labels provided, create uniform partitioning
            return self._uniform_partition(dataset)
        
        # Group indices by label
        label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Initialize client indices
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Distribute each label's indices according to Dirichlet distribution
        for label, indices in label_to_indices.items():
            # Sample proportions for this label across clients
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Calculate how many samples each client gets for this label
            label_count = len(indices)
            client_counts = (proportions * label_count).astype(int)
            
            # Adjust for rounding errors
            diff = label_count - sum(client_counts)
            for i in range(diff):
                client_counts[i] += 1
            
            # Assign indices to clients
            start = 0
            for client_id, count in enumerate(client_counts):
                end = start + count
                client_indices[client_id].extend(indices[start:end])
                start = end
        
        # Create Subset objects
        subsets = [Subset(dataset, indices) for indices in client_indices]
        return subsets
    
    def _uniform_partition(self, dataset: Dataset) -> List[Subset]:
        """Create uniform (IID) partitioning when no labels are available."""
        n = len(dataset)
        indices = np.random.permutation(n)
        splits = np.array_split(indices, self.num_clients)
        subsets = [Subset(dataset, indices.tolist()) for indices in splits]
        return subsets
    
    def visualize_distribution(
        self, 
        dataset: Dataset, 
        client_subsets: List[Subset],
        labels: Optional[List[Any]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize the data distribution across clients.
        
        Args:
            dataset: Original dataset
            client_subsets: List of client data subsets
            labels: List of labels for each data point
            save_path: Path to save the visualization (if None, shows plot)
        """
        if labels is None:
            print("Cannot visualize distribution without labels")
            return
        
        # Get label distribution for each client
        client_label_dists = []
        for subset in client_subsets:
            subset_labels = [labels[i] for i in subset.indices]
            dist = Counter(subset_labels)
            client_label_dists.append(dist)
        
        # Get all unique labels
        all_labels = sorted(set(labels))
        
        # Create matrix for heatmap
        dist_matrix = np.zeros((len(client_subsets), len(all_labels)))
        for i, dist in enumerate(client_label_dists):
            for j, label in enumerate(all_labels):
                dist_matrix[i, j] = dist.get(label, 0)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            dist_matrix, 
            annot=True, 
            fmt='d',
            cmap='YlOrRd',
            xticklabels=all_labels,
            yticklabels=[f'Client {i}' for i in range(len(client_subsets))]
        )
        plt.title(f'Non-IID Data Distribution (α={self.alpha})')
        plt.xlabel('Label')
        plt.ylabel('Client')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()


def earth_mover_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        EMD score
    """
    # Ensure distributions are normalized
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Compute cumulative distributions
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    
    # EMD is the L1 distance between CDFs
    emd = np.sum(np.abs(cdf_p - cdf_q))
    return emd


def analyze_heterogeneity(
    client_data_list: List[List[Any]], 
    get_label_fn: callable,
    get_vocab_fn: Optional[callable] = None
) -> HeterogeneityMetrics:
    """
    Analyze heterogeneity across client data distributions.
    
    Args:
        client_data_list: List of client data (each element is a list of data points)
        get_label_fn: Function to extract label from a data point
        get_vocab_fn: Function to extract vocabulary from a data point (optional)
        
    Returns:
        HeterogeneityMetrics object containing analysis results
    """
    num_clients = len(client_data_list)
    
    # Get label distribution for each client
    client_label_dists = []
    all_labels = set()
    
    for client_data in client_data_list:
        labels = [get_label_fn(data_point) for data_point in client_data]
        label_counts = Counter(labels)
        client_label_dists.append(label_counts)
        all_labels.update(labels)
    
    all_labels = sorted(list(all_labels))
    num_labels = len(all_labels)
    
    # Convert to probability distributions
    label_distributions = []
    for dist in client_label_dists:
        prob_dist = np.array([dist.get(label, 0) for label in all_labels], dtype=float)
        if np.sum(prob_dist) > 0:
            prob_dist = prob_dist / np.sum(prob_dist)
        label_distributions.append(prob_dist)
    
    # Compute EMD between each client and the global distribution
    global_dist = np.mean(label_distributions, axis=0)
    emd_scores = []
    for client_dist in label_distributions:
        emd = earth_mover_distance(client_dist, global_dist)
        emd_scores.append(emd)
    
    # Compute label skew (KL divergence from uniform)
    uniform_dist = np.ones(num_labels) / num_labels
    label_skew = []
    for client_dist in label_distributions:
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        client_dist_smooth = client_dist + eps
        uniform_dist_smooth = uniform_dist + eps
        client_dist_smooth = client_dist_smooth / np.sum(client_dist_smooth)
        uniform_dist_smooth = uniform_dist_smooth / np.sum(uniform_dist_smooth)
        
        # KL divergence
        kl_div = np.sum(client_dist_smooth * np.log(client_dist_smooth / uniform_dist_smooth))
        label_skew.append({label: float(client_dist_smooth[i]) for i, label in enumerate(all_labels)})
    
    # Compute vocabulary overlap (if function provided)
    vocab_overlap = []
    if get_vocab_fn is not None:
        client_vocabs = []
        for client_data in client_data_list:
            vocab_set = set()
            for data_point in client_data:
                vocab_set.update(get_vocab_fn(data_point))
            client_vocabs.append(vocab_set)
        
        # Compute Jaccard similarity between each client and the union
        union_vocab = set().union(*client_vocabs) if client_vocabs else set()
        for client_vocab in client_vocabs:
            if len(union_vocab) > 0:
                intersection = len(client_vocab & union_vocab)
                union = len(client_vocab | union_vocab)
                jaccard = intersection / union if union > 0 else 0.0
            else:
                jaccard = 0.0
            vocab_overlap.append(jaccard)
    else:
        vocab_overlap = [0.0] * num_clients
    
    return HeterogeneityMetrics(
        emd_scores=emd_scores,
        label_skew=label_skew,
        vocab_overlap=vocab_overlap,
        num_clients=num_clients
    )


class FedProxRegularizer:
    """
    Implements proximal term to handle non-IID data (FedProx algorithm).
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx regularizer.
        
        Args:
            mu: Proximal term coefficient (higher = stronger regularization)
        """
        self.mu = mu
    
    def compute_proximal_loss(
        self, 
        local_model: nn.Module, 
        global_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute proximal loss: (mu/2) * ||w - w_t||^2
        
        Args:
            local_model: Current local model
            global_model: Global model from server
            
        Returns:
            Proximal loss term
        """
        proximal_loss = 0.0
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            proximal_loss += torch.sum((local_param - global_param) ** 2)
        proximal_loss = (self.mu / 2.0) * proximal_loss
        return proximal_loss


if __name__ == "__main__":
    # Test DirichletPartitioner
    print("Testing DirichletPartitioner...")
    
    # Create a simple dataset with labels
    class SimpleDataset(Dataset):
        def __init__(self, size=100, num_classes=5):
            self.data = list(range(size))
            self.labels = [i % num_classes for i in range(size)]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = SimpleDataset(size=100, num_classes=3)
    labels = [label for _, label in dataset]
    
    # Test with different alpha values
    for alpha in [0.1, 0.5, 1.0]:
        print(f"\nTesting with alpha={alpha}")
        partitioner = DirichletPartitioner(num_clients=5, alpha=alpha, seed=42)
        client_subsets = partitioner.partition(dataset, labels)
        
        # Print distribution stats
        for i, subset in enumerate(client_subsets):
            subset_labels = [labels[idx] for idx in subset.indices]
            dist = Counter(subset_labels)
            print(f"  Client {i}: {dict(dist)}")
    
    # Test analyze_heterogeneity
    print("\n\nTesting analyze_heterogeneity...")
    client_data = [[subset.indices for subset in client_subsets]]  # Simplified
    # Actually, let's create proper client data
    client_data_list = [list(subset.indices) for subset in client_subsets]
    
    def get_label_from_idx(idx):
        return labels[idx]
    
    metrics = analyze_heterogeneity(client_data_list, get_label_from_idx)
    print(f"EMD scores: {[f'{score:.4f}' for score in metrics.emd_scores]}")
    print(f"Average EMD: {np.mean(metrics.emd_scores):.4f}")
    
    # Test FedProxRegularizer
    print("\n\nTesting FedProxRegularizer...")
    import torch.nn as nn
    
    # Simple models
    local_model = nn.Linear(10, 1)
    global_model = nn.Linear(10, 1)
    
    # Make them different
    with torch.no_grad():
        local_model.weight += 0.1
        local_model.bias += 0.1
    
    fedprox = FedProxRegularizer(mu=0.01)
    proximal_loss = fedprox.compute_proximal_loss(local_model, global_model)
    print(f"Proximal loss: {proximal_loss.item():.6f}")
    
    # Test with same models (should be zero)
    global_model.weight.copy_(local_model.weight)
    global_model.bias.copy_(local_model.bias)
    proximal_loss_same = fedprox.compute_proximal_loss(local_model, global_model)
    print(f"Proximal loss (same models): {proximal_loss_same.item():.6f}")