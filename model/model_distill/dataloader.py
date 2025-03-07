import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np

class ProteinDataset(Dataset):
    """Dataset for protein sequences from FASTA files"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 512,
        file_pattern: str = "*.fasta",
    ):
        """
        Initialize a protein sequence dataset
        
        Args:
            data_path: Path to directory containing FASTA files
            tokenizer: Tokenizer to use for tokenization
            max_length: Maximum sequence length
            file_pattern: Pattern to match FASTA files
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load sequences from FASTA files
        self.sequences = []
        self.headers = []
        
        # Check if directory or specific file
        if self.data_path.is_dir():
            for fasta_file in self.data_path.glob(file_pattern):
                self._load_fasta(fasta_file)
        else:
            self._load_fasta(self.data_path)
        
        print(f"Loaded {len(self.sequences)} sequences from {self.data_path}")
    
    def _load_fasta(self, fasta_path: Path):
        """Load sequences from a FASTA file"""
        with open(fasta_path, "r") as f:
            current_header = None
            current_sequence = ""
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_header and current_sequence:
                        self.headers.append(current_header)
                        self.sequences.append(current_sequence)
                    
                    # Start new sequence
                    current_header = line
                    current_sequence = ""
                else:
                    current_sequence += line
            
            # Add the last sequence
            if current_header and current_sequence:
                self.headers.append(current_header)
                self.sequences.append(current_sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single sequence"""
        sequence = self.sequences[idx]
        header = self.headers[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        
        # Truncate if needed
        if len(tokens) > self.max_length - 2:  # -2 for special tokens
            tokens = tokens[:self.max_length - 2]
        
        # Convert to token IDs and add special tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        encoded = self.tokenizer.prepare_for_model(
            token_ids,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract the needed tensors
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sequence": sequence,
            "header": header
        }


class MLMCollator:
    """Collator for masked language modeling"""
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        max_length: int = 512
    ):
        """
        Initialize collator
        
        Args:
            tokenizer: Tokenizer to use for tokenization
            mlm_probability: Probability of masking a token
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples
        
        Args:
            examples: List of examples
            
        Returns:
            Batch with masked tokens and labels
        """
        # Collate inputs
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        
        # Create masked inputs and labels
        inputs, labels = self.mask_tokens(input_ids, attention_mask)
        
        # Return batch
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels,
            "sequences": [example["sequence"] for example in examples],
            "headers": [example["header"] for example in examples]
        }
    
    def mask_tokens(
        self, 
        inputs: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling
        
        Args:
            inputs: Batch of input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (masked inputs, labels)
        """
        # Clone inputs
        labels = inputs.clone()
        
        # We sample a few tokens in each sequence for MLM training
        # with probability mlm_probability (15% in BERT)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Don't mask padding tokens
        padding_mask = attention_mask.eq(0)
        
        # Create combined mask
        combined_mask = special_tokens_mask | padding_mask
        probability_matrix.masked_fill_(combined_mask, value=0.0)
        
        # Sample mask indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for unmasked tokens to -100
        labels[~masked_indices] = -100
        
        # Create masks for different token replacements
        # 80% of the time, replace with [MASK]
        # 10% of the time, replace with random token
        # 10% of the time, keep original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # Indices to replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time, keep the original tokens (10% of masked tokens)
        
        return inputs, labels 