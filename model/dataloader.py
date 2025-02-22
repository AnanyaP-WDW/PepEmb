import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from tokenizer import Tokenizer

class PeptideDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_length: int = 60,
        mask_prob: float = 0.15,
        is_train: bool = True
    ):
        """
        Initialize PeptideDataset.
        
        Args:
            data_path (str): Path to the CSV file containing peptide data
            tokenizer (Tokenizer): Tokenizer instance for encoding sequences
            max_length (int): Maximum sequence length (including special tokens)
            mask_prob (float): Probability of masking tokens during training
            is_train (bool): Whether this is training data (affects masking)
        """
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.is_train = is_train
        
        # Extract scaled features
        self.feature_columns = [col for col in self.df.columns if col.endswith('_scaled')]

    def __len__(self) -> int:
        return len(self.df)

    def _extract_features(self, row: pd.Series) -> torch.Tensor:
        """
        Extract scaled features.
        
        Args:
            row (pd.Series): Row from DataFrame containing scaled features
            
        Returns:
            torch.Tensor: Feature vector [num_features]
        """
        features = row[self.feature_columns].values
        return torch.tensor(features.astype(np.float32), dtype=torch.float32)

    def _apply_masking(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Apply BERT-style masking to input tokens."""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # Use -100 for PyTorch's ignore_index
        
        for i in range(len(tokens)):
            # Skip special tokens
            if tokens[i] in [self.tokenizer.amino_acids['<PAD>'], 
                           self.tokenizer.amino_acids['<BOS>'],
                           self.tokenizer.amino_acids['<EOS>']]:
                continue
                
            if np.random.random() < self.mask_prob:
                labels[i] = tokens[i]  # Save original token as label
                prob = np.random.random()
                
                if prob < 0.8:  # 80% mask
                    masked_tokens[i] = self.tokenizer.amino_acids['<MASK>']
                elif prob < 0.9:  # 10% random
                    masked_tokens[i] = np.random.randint(1, 21)  # Random amino acid
                # else: 10% unchanged - keep original token
                
        return masked_tokens, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Tokenized and masked sequence
                - labels: Labels for masked positions
                - attention_mask: Mask for padding tokens
                - features: Scaled features
        """
        row = self.df.iloc[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(row['sequence'])
        tokens = self.tokenizer.pad_sequence(tokens, self.max_length)
        
        # Apply masking for both training and validation
        tokens, labels = self._apply_masking(tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if t != self.tokenizer.amino_acids['<PAD>'] else 0 
                         for t in tokens]
        
        # Extract scaled features
        features = self._extract_features(row)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'features': features
        }

def create_peptide_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: Tokenizer,
    batch_size: int = 8,
    max_length: int = 60,
    mask_prob: float = 0.15,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_path (str): Path to training data CSV
        val_path (str): Path to validation data CSV
        tokenizer (Tokenizer): Tokenizer instance
        batch_size (int): Batch size for dataloaders
        max_length (int): Maximum sequence length
        mask_prob (float): Probability of masking tokens
        num_workers (int): Number of worker processes
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    train_dataset = PeptideDataset(
        train_path, tokenizer, max_length, mask_prob, is_train=True
    )
    val_dataset = PeptideDataset(
        val_path, tokenizer, max_length, mask_prob, is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
