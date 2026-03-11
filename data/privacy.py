"""
Privacy-preserving data preprocessing for federated learning.
Implements PII removal, differential privacy, and secure aggregation placeholders.
"""

import re
import torch
import numpy as np
from typing import List, Tuple, Optional
import string


def sanitize_text(text: str) -> str:
    """
    Remove personally identifiable information (PII) from text.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text with PII removed/replaced
    """
    if not isinstance(text, str):
        return ""
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[EMAIL]', text)
    
    # Remove phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # XXX-XXX-XXXX
        r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (XXX) XXX-XXXX
        r'\b\d{3}\s\d{3}\s\d{4}\b',  # XXX XXX XXXX
        r'\b\+?\d{1,3}[-.]?\d{1,4}[-.]?\d{1,4}[-.]?\d{1,4}\b',  # International
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, '[PHONE]', text)
    
    # Remove URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '[URL]', text)
    
    # Remove IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    text = re.sub(ip_pattern, '[IP_ADDRESS]', text)
    
    # Simple name replacement (basic NER - replace common patterns)
    # This is a simplified version - in practice would use a proper NER model
    name_patterns = [
        r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # First Last
        r'\b[A-Z][a-z]+\s[A-Z]\.\s[A-Z][a-z]+\b',  # First M. Last
    ]
    for pattern in name_patterns:
        text = re.sub(pattern, '[NAME]', text)
    
    return text


def differential_privacy_noise(
    gradients: torch.Tensor, 
    epsilon: float = 1.0, 
    delta: float = 1e-5,
    clip_norm: float = 1.0
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise for (ε,δ)-differential privacy.
    
    Args:
        gradients: Gradient tensor to add noise to
        epsilon: Privacy budget parameter (smaller = more private)
        delta: Privacy budget parameter (probability of failure)
        clip_norm: Maximum L2 norm for gradient clipping
        
    Returns:
        Noised gradients tensor
    """
    # Clone gradients to avoid modifying original
    noised_gradients = gradients.clone()
    
    # Clip gradients
    grad_norm = torch.norm(noised_gradients, p=2)
    if grad_norm > clip_norm:
        noised_gradients = noised_gradients * (clip_norm / grad_norm)
    
    # Calculate noise scale
    # For Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    sensitivity = clip_norm  # L2 sensitivity after clipping
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    # Add Gaussian noise
    noise = torch.normal(mean=0.0, std=sigma, size=gradients.shape, device=gradients.device)
    noised_gradients = noised_gradients + noise
    
    return noised_gradients


class SecureAggregation:
    """
    Placeholder for secure aggregation protocol implementation.
    Documents the protocol for secure aggregation in federated learning.
    """
    
    def __init__(self, num_clients: int, security_parameter: int = 128):
        """
        Initialize secure aggregation.
        
        Args:
            num_clients: Number of clients participating in aggregation
            security_parameter: Security parameter for cryptographic operations
        """
        self.num_clients = num_clients
        self.security_parameter = security_parameter
        self.masks = {}
        self.unmasked_values = {}
    
    def generate_mask(self, client_id: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate a random mask for a client.
        
        Args:
            client_id: ID of the client
            shape: Shape of the tensor to mask
            
        Returns:
            Random mask tensor
        """
        # In a real implementation, this would use cryptographic techniques
        # For now, we'll generate a deterministic mask based on client_id
        torch.manual_seed(client_id * 42)  # Deterministic for testing
        mask = torch.randn(shape)
        self.masks[client_id] = mask
        return mask
    
    def apply_mask(self, client_id: int, value: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to a client's value.
        
        Args:
            client_id: ID of the client
            value: Value to mask
            
        Returns:
            Masked value
        """
        if client_id not in self.masks:
            self.generate_mask(client_id, value.shape)
        
        masked_value = value + self.masks[client_id]
        self.unmasked_values[client_id] = value  # Store original for verification
        return masked_value
    
    def aggregate_masked(self, masked_values: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate masked values from clients.
        
        Args:
            masked_values: List of masked values from clients
            
        Returns:
            Aggregated value (sum of masked values)
        """
        if len(masked_values) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} values, got {len(masked_values)}")
        
        # Sum all masked values
        aggregated = torch.stack(masked_values).sum(dim=0)
        
        # In a real implementation, the masks would cancel out
        # For our placeholder, we'll subtract the sum of known masks
        # This is only correct if we know all client IDs
        mask_sum = torch.zeros_like(aggregated)
        for client_id in range(self.num_clients):
            if client_id in self.masks:
                mask_sum += self.masks[client_id]
        
        # Remove the mask contribution
        unmasked_aggregated = aggregated - mask_sum
        return unmasked_aggregated
    
    def verify_aggregation(self, original_values: List[torch.Tensor], aggregated: torch.Tensor) -> bool:
        """
        Verify that aggregation was performed correctly.
        
        Args:
            original_values: Original values from clients
            aggregated: Allegedly aggregated value
            
        Returns:
            True if verification passes
        """
        expected_sum = torch.stack(original_values).sum(dim=0)
        return torch.allclose(aggregated, expected_sum, atol=1e-6)


if __name__ == "__main__":
    # Test sanitize_text
    test_text = """
    Contact me at john.doe@example.com or call 555-123-4567.
    Visit https://example.com for more info.
    My IP is 192.168.1.1 and my name is John Smith.
    """
    sanitized = sanitize_text(test_text)
    print("Original:", test_text)
    print("Sanitized:", sanitized)
    
    # Test differential privacy
    grads = torch.tensor([1.0, 2.0, 3.0])
    noised = differential_privacy_noise(grads, epsilon=1.0, delta=1e-5)
    print("\nOriginal gradients:", grads)
    print("Noised gradients:", noised)
    
    # Test SecureAggregation
    sa = SecureAggregation(num_clients=3)
    values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
    masked = [sa.apply_mask(i, v) for i, v in enumerate(values)]
    aggregated = sa.aggregate_masked(masked)
    expected = torch.tensor([9.0, 12.0])  # 1+3+5, 2+4+6
    print(f"\nSecure aggregation test:")
    print(f"Aggregated: {aggregated}")
    print(f"Expected: {expected}")
    print(f"Correct: {torch.allclose(aggregated, expected)}")