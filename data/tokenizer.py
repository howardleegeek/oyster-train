"""
Tokenizer wrapper for Qwen2.5 model supporting Chinese and English text.
Provides encoding/decoding functionality and training pair creation.
"""

from typing import List, Dict, Any, Iterator
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset


class InputOutputPair:
    """Represents an input-output pair for instruction tuning."""
    
    def __init__(self, input_ids: List[int], output_ids: List[int]):
        self.input_ids = input_ids
        self.output_ids = output_ids
    
    def __repr__(self):
        return f"InputOutputPair(input_len={len(self.input_ids)}, output_len={len(self.output_ids)})"


class OysterTokenizer:
    """
    Wrapper around Qwen2.5 tokenizer for handling bilingual text.
    
    Attributes:
        tokenizer: The underlying HuggingFace tokenizer
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_length: int = 256):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: Name or path of the pretrained model
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        
        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            max_length: Maximum length (uses instance default if None)
            
        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self.max_length
            
        encoded = self.tokenizer.encode(
            text, 
            max_length=max_length, 
            truncation=True,
            padding=False
        )
        return encoded
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def create_training_pairs(self, text: str) -> List[InputOutputPair]:
        """
        Create instruction/response pairs from text for instruction tuning.
        For language modeling, creates sliding window chunks.
        
        Args:
            text: Input text to create pairs from
            
        Returns:
            List of InputOutputPair objects
        """
        # Simple heuristic: split by common instruction/response patterns
        # In practice, this would be more sophisticated
        pairs = []
        
        # Try to split by common delimiters that indicate instruction/response
        sections = text.split('\n\n')
        
        for i in range(0, len(sections)-1, 2):  # Process in pairs
            if i+1 < len(sections):
                instruction = sections[i].strip()
                response = sections[i+1].strip()
                
                if instruction and response:
                    input_ids = self.encode(instruction)
                    output_ids = self.encode(response)
                    pairs.append(InputOutputPair(input_ids, output_ids))
        
        # If no pairs found, fall back to language modeling approach
        if not pairs:
            tokens = self.encode(text)
            # Create sliding window chunks for language modeling
            for i in range(0, len(tokens) - self.max_length//2, self.max_length//2):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) >= self.max_length//2:  # Minimum chunk size
                    # For language modeling, input is chunk[:-1], output is chunk[1:]
                    input_ids = chunk[:-1]
                    output_ids = chunk[1:]
                    pairs.append(InputOutputPair(input_ids, output_ids))
        
        return pairs
    
    def batch_encode(self, texts: List[str], batch_size: int = 4) -> DataLoader:
        """
        Encode a batch of texts and return a DataLoader.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader yielding batches of token IDs
        """
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoded = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                return encoded.squeeze(0)  # Remove batch dimension
        
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def save_vocab(self, save_directory: str):
        """
        Save the tokenizer vocabulary for on-device use.
        
        Args:
            save_directory: Directory to save the vocabulary files
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)


if __name__ == "__main__":
    # Simple test
    tokenizer = OysterTokenizer()
    test_text = "Hello, 你好！This is a test. 这是一个测试。"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {test_text == decoded}")