import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import gc

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute accuracy only on masked tokens"""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & mask
    total = mask.sum()
    return (correct.sum() / total).item() if total > 0 else 0.0

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    grad_accum_steps: int = 1
) -> Dict[str, float]:
    model.train()
    total_loss = 0
    total_accuracy = 0
    steps = 0

    clear_memory()
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        
        # Forward pass
        outputs = model(input_ids, features, attention_mask)
        
        # Compute loss on all tokens (CrossEntropyLoss will automatically ignore -100)
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)),  # Reshape to [batch_size * seq_len, vocab_size]
            labels.view(-1)  # Reshape to [batch_size * seq_len]
        )
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # Compute accuracy
        accuracy = compute_accuracy(outputs, labels, attention_mask)
        
        # Update metrics
        total_loss += loss.item() * grad_accum_steps
        total_accuracy += accuracy
        steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / steps,
            'accuracy': total_accuracy / steps
        })
    
    return {
        'loss': total_loss / steps,
        'accuracy': total_accuracy / steps
    }

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    steps = 0

    clear_memory()
    
    progress_bar = tqdm(val_loader, desc="Evaluating")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        
        # Forward pass
        outputs = model(input_ids, features, attention_mask)
        
        # Compute loss on all tokens (CrossEntropyLoss will automatically ignore -100)
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)),  # Reshape to [batch_size * seq_len, vocab_size]
            labels.view(-1)  # Reshape to [batch_size * seq_len]
        )
        
        # Compute accuracy only on masked tokens
        mask = labels != -100
        accuracy = compute_accuracy(outputs, labels, mask)
        
        # Update metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / steps,
            'accuracy': total_accuracy / steps
        })
    
    return {
        'loss': total_loss / steps,
        'accuracy': total_accuracy / steps
    }

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 10,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_dir: str = 'checkpoints',
    grad_accum_steps: int = 1,
    patience: int = 7
) -> Dict[str, list]:
    
    print(f"device:{device}")
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Initialize metrics tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, grad_accum_steps
        )
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, checkpoint_dir / 'best_model.pt')
        
        # Save history
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print("\nEarly stopping triggered!")
            break
    
    return history

def clear_memory():
    """Clear GPU and CPU memory, cache, and garbage collect"""
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # Clear CPU memory
    gc.collect()
    
    # Optional: Clear CUDA memory fragments
    if torch.cuda.is_available():
        torch.cuda.synchronize()



if __name__ == "__main__":
    from architecture import CustomPeptideEncoder, MLMPeptideModel
    from dataloader import create_peptide_dataloaders
    from tokenizer import Tokenizer
    
    # Initialize model parameters
    vocab_size = 25  # 20 amino acids + 5 special tokens
    d_model = 64
    num_heads = 4
    num_layers = 2
    ff_dim = 256
    max_seq_length = 60
    num_features = 4
    batch_size = 4
    
    # Create model with MLM head
    encoder = CustomPeptideEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_seq_length=max_seq_length,
        num_features=num_features
    )
    model = MLMPeptideModel(encoder, vocab_size)
    
    # Create tokenizer and dataloaders
    tokenizer = Tokenizer()
    train_loader, val_loader = create_peptide_dataloaders(
        train_path='../peptide_data/train.csv',
        val_path='../peptide_data/val.csv',
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seq_length
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10,  # number of epochs
        eta_min=1e-6
    )
    
    # Train model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        checkpoint_dir='checkpoints',
        grad_accum_steps=4,
        patience=7
    )

    clear_memory()
