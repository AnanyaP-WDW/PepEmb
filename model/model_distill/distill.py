import os
import sys
import argparse
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients, visualization
from tqdm import tqdm
import wandb

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_distill.dataloader import ProteinDataset, MLMCollator
from model_distill.architecture import ProteinTransformerStudent

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str = "Rostlab/prot_bert",
        student_config: Dict[str, Any] = None,
        data_path: str = "data/unref_50",
        output_dir: str = "model_distill/checkpoints",
        mask_probability: float = 0.15,
        distill_weight: float = 0.5,
        task_weight: float = 0.5,
        temperature: float = 1.0,
        batch_size: int = 32,
        max_seq_len: int = 512,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        num_epochs: int = 10,
        grad_accumulation_steps: int = 1,
        device: str = None,
        use_wandb: bool = False,
        fp16: bool = False,
        use_8bit: bool = False,
        save_every: int = 10000,
        eval_every: int = 1000,
        log_every: int = 100,
    ):
        """
        Initialize the distillation trainer
        
        Args:
            teacher_model_name: HuggingFace model name for teacher model
            student_config: Configuration for student model
            data_path: Path to training data
            output_dir: Directory to save checkpoints
            mask_probability: Probability of masking tokens for MLM
            distill_weight: Weight for distillation loss
            task_weight: Weight for task loss (MLM)
            temperature: Temperature for distillation
            batch_size: Batch size for training
            max_seq_len: Maximum sequence length
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            num_epochs: Number of training epochs
            grad_accumulation_steps: Number of steps to accumulate gradients
            device: Device to use (cpu, cuda, cuda:0, etc.)
            use_wandb: Whether to use W&B for logging
            fp16: Whether to use mixed precision training
            use_8bit: Whether to use 8-bit quantization (for memory efficiency)
            save_every: Save checkpoint every N steps
            eval_every: Evaluate every N steps
            log_every: Log metrics every N steps
        """
        self.mask_probability = mask_probability
        self.distill_weight = distill_weight
        self.task_weight = task_weight
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.use_wandb = use_wandb
        self.fp16 = fp16
        self.use_8bit = use_8bit
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data paths
        self.data_path = Path(data_path)
        
        # Load tokenizer from teacher model
        self.tokenizer = BertTokenizer.from_pretrained(teacher_model_name)
        
        # Set student config defaults if not provided
        if student_config is None:
            self.student_config = {
                "vocab_size": len(self.tokenizer),
                "d_model": 384,  # Smaller than teacher
                "num_heads": 6,
                "num_layers": 4,  # Fewer layers than teacher
                "d_ff": 1536,
                "max_seq_len": max_seq_len,
                "dropout": 0.1,
            }
        else:
            self.student_config = student_config
            
        # Initialize models
        self._init_models(teacher_model_name)
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.fp16 else None
        
        # Initialize metrics
        self.best_eval_loss = float('inf')
        self.global_step = 0
        
    def _init_models(self, teacher_model_name: str):
        """Initialize teacher and student models"""
        print("Initializing models...")
        
        # Load teacher model
        if self.use_8bit:
            # Use 8-bit quantization for memory efficiency
            from transformers import AutoModelForMaskedLM, BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.teacher = AutoModelForMaskedLM.from_pretrained(
                teacher_model_name,
                quantization_config=quantization_config,
            )
        else:
            self.teacher = BertForMaskedLM.from_pretrained(teacher_model_name)
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
        self.teacher.to(self.device)
        
        # Ensure vocab_size is in student_config
        if "vocab_size" not in self.student_config:
            self.student_config["vocab_size"] = len(self.tokenizer)
            
        # Initialize student model with RoPE embeddings
        self.student = ProteinTransformerStudent(**self.student_config)
        self.student.to(self.device)
        
        # Count parameters
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        print(f"Teacher model parameters: {teacher_params:,}")
        print(f"Student model parameters: {student_params:,}")
        print(f"Compression ratio: {teacher_params / student_params:.2f}x")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        # Initialize optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() 
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.student.named_parameters() 
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def _init_data_loaders(self):
        """Initialize data loaders for training and evaluation"""
        # Create datasets
        train_dataset = ProteinDataset(
            data_path=self.data_path / "train",
            tokenizer=self.tokenizer,
            max_length=self.max_seq_len,
        )
        
        val_dataset = ProteinDataset(
            data_path=self.data_path / "val",
            tokenizer=self.tokenizer,
            max_length=self.max_seq_len,
        )
        
        # Create collators
        train_collator = MLMCollator(
            tokenizer=self.tokenizer,
            mlm_probability=self.mask_probability
        )
        
        val_collator = MLMCollator(
            tokenizer=self.tokenizer,
            mlm_probability=self.mask_probability
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=train_collator,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=val_collator,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Create scheduler based on total steps
        total_steps = len(self.train_loader) * self.num_epochs // self.grad_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        print(f"Training on {len(train_dataset):,} sequences")
        print(f"Validating on {len(val_dataset):,} sequences")
    
    def compute_distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute distillation and task losses
        
        Args:
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            labels: True labels with masked tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (combined_loss, distillation_loss, task_loss)
        """
        # Create mask for only calculating loss on masked tokens
        masked_positions = (labels != -100)
        
        # Calculate task loss (cross entropy on MLM)
        task_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
        
        # Apply temperature scaling for distillation
        scaled_teacher_logits = teacher_logits / self.temperature
        scaled_student_logits = student_logits / self.temperature
        
        # Calculate distillation loss (KL divergence)
        teacher_probs = F.softmax(scaled_teacher_logits, dim=-1)
        
        # Only compute KL divergence on masked tokens
        distill_loss = F.kl_div(
            F.log_softmax(scaled_student_logits, dim=-1),
            teacher_probs,
            reduction="none"
        )
        
        # Average over masked positions and batch
        distill_loss = distill_loss.sum(-1)
        
        # Apply mask for masked tokens only
        distill_loss = (distill_loss * masked_positions).sum() / (masked_positions.sum() + 1e-8)
        
        # Combine losses
        combined_loss = self.distill_weight * distill_loss + self.task_weight * task_loss
        
        return combined_loss, distill_loss, task_loss
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass through teacher model (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits
        
        # Forward pass through student model
        if self.fp16:
            with autocast():
                # Create attention mask in the format expected by the model
                # The format should be [batch_size, seq_len, seq_len] for the student model
                seq_len = input_ids.size(1)
                extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
                
                student_logits = self.student(
                    x=input_ids,
                    mask=extended_attention_mask
                )
                
                # Calculate losses
                loss, distill_loss, task_loss = self.compute_distillation_loss(
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Step optimizer with gradient accumulation
            if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            # Forward pass
            # Create attention mask in the format expected by the model
            # The format should be [batch_size, seq_len, seq_len] for the student model
            seq_len = input_ids.size(1)
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            
            student_logits = self.student(
                x=input_ids,
                mask=extended_attention_mask
            )
            
            # Calculate losses
            loss, distill_loss, task_loss = self.compute_distillation_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=labels,
                attention_mask=attention_mask,
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Step optimizer with gradient accumulation
            if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Calculate metrics
        with torch.no_grad():
            # Get predictions
            predictions = torch.argmax(student_logits, dim=-1)
            
            # Calculate accuracy (only on masked tokens)
            mask = (labels != -100)
            correct = (predictions[mask] == labels[mask]).float()
            accuracy = correct.sum() / mask.sum()
        
        # Return metrics
        return {
            "loss": loss.item() * self.grad_accumulation_steps,
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item(),
            "accuracy": accuracy.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader = None) -> Dict[str, float]:
        """Evaluate the model on the validation set"""
        if val_loader is None:
            val_loader = self.val_loader
        
        self.student.eval()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass through teacher model
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits
            
            # Create attention mask in the format expected by the model
            # The format should be [batch_size, seq_len, seq_len] for the student model
            seq_len = input_ids.size(1)
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Forward pass through student model
            student_logits = self.student(
                x=input_ids,
                mask=extended_attention_mask
            )
            
            # Calculate losses
            loss, distill_loss, task_loss = self.compute_distillation_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=labels,
                attention_mask=attention_mask,
            )
            
            # Get predictions
            predictions = torch.argmax(student_logits, dim=-1)
            
            # Calculate accuracy (only on masked tokens)
            mask = (labels != -100)
            correct = (predictions[mask] == labels[mask]).float()
            accuracy = correct.sum() / mask.sum()
            
            # Update metrics
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        # Calculate average metrics
        avg_metrics = {
            "val_loss": total_loss / num_batches,
            "val_distill_loss": total_distill_loss / num_batches,
            "val_task_loss": total_task_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches,
        }
        
        self.student.train()
        return avg_metrics
    
    def generate_saliency_map(self, sequence: str, tokenizer: BertTokenizer = None):
        """
        Generate saliency map for a given sequence
        
        Args:
            sequence: Protein sequence to analyze
            tokenizer: Optional tokenizer (uses self.tokenizer if None)
        
        Returns:
            Visualization of attention weights and saliency map
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        # Tokenize input
        tokens = tokenizer.tokenize(sequence)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.prepare_for_model(input_ids, max_length=self.max_seq_len, truncation=True)["input_ids"]
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # Create a simple attention mask (all 1s for actual tokens)
        attention_mask = torch.ones_like(input_tensor)
        
        # Create attention mask in the correct format
        seq_len = input_tensor.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get attention weights and logits
        self.student.eval()
        with torch.no_grad():
            logits, attentions = self.student(
                x=input_tensor, 
                mask=extended_attention_mask,
                return_attention=True
            )
            
        # Plot attention weights
        fig, axes = plt.subplots(1, len(attentions), figsize=(5*len(attentions), 5))
        if len(attentions) == 1:
            axes = [axes]
            
        for i, attn in enumerate(attentions):
            # Average over heads
            attn = attn.mean(dim=1).squeeze(0).cpu().numpy()
            
            # Plot attention matrix
            im = axes[i].imshow(attn, cmap='viridis')
            axes[i].set_title(f"Layer {i+1} Attention")
            axes[i].set_xlabel("Sequence Position")
            axes[i].set_ylabel("Sequence Position")
            
            # Show amino acid labels on x and y axes
            if len(tokens) <= 50:  # Only show labels for shorter sequences
                axes[i].set_xticks(range(len(tokens)))
                axes[i].set_xticklabels(tokens, rotation=90)
                axes[i].set_yticks(range(len(tokens)))
                axes[i].set_yticklabels(tokens)
                
            fig.colorbar(im, ax=axes[i])
        
        fig.tight_layout()
        return fig
    
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None, metrics: Dict[str, float] = None):
        """Save model checkpoint"""
        if path is None:
            path = self.output_dir / f"checkpoint-{self.global_step}"
            
        os.makedirs(path, exist_ok=True)
        
        # Save model
        torch.save(self.student.state_dict(), os.path.join(path, "model.pt"))
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, os.path.join(path, "optimizer.pt"))
        
        # Save training configuration
        config = {
            "mask_probability": self.mask_probability,
            "distill_weight": self.distill_weight,
            "task_weight": self.task_weight,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "num_epochs": self.num_epochs,
            "student_config": self.student_config,
        }
        
        if metrics is not None:
            config["metrics"] = metrics
            
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop"""
        self._init_data_loaders()
        
        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="protein-distillation",
                    config={
                        "mask_probability": self.mask_probability,
                        "distill_weight": self.distill_weight,
                        "task_weight": self.task_weight,
                        "temperature": self.temperature,
                        "batch_size": self.batch_size,
                        "max_seq_len": self.max_seq_len,
                        "learning_rate": self.learning_rate,
                        "warmup_steps": self.warmup_steps,
                        "num_epochs": self.num_epochs,
                        "grad_accumulation_steps": self.grad_accumulation_steps,
                        "student_config": self.student_config,
                        "device": str(self.device),
                        "fp16": self.fp16,
                    }
                )
            except ImportError:
                print("wandb not installed, skipping wandb logging")
                self.use_wandb = False
        
        # Training loop
        self.student.train()
        self.optimizer.zero_grad()
        
        print("Starting training...")
        total_train_loss = 0
        log_loss = 0
        log_distill_loss = 0
        log_task_loss = 0
        log_accuracy = 0
        log_steps = 0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            epoch_iterator = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            for batch in epoch_iterator:
                metrics = self.train_step(batch)
                
                # Update metrics
                total_train_loss += metrics["loss"]
                log_loss += metrics["loss"]
                log_distill_loss += metrics["distill_loss"]
                log_task_loss += metrics["task_loss"]
                log_accuracy += metrics["accuracy"]
                log_steps += 1
                
                # Log metrics
                if (self.global_step + 1) % self.log_every == 0 and log_steps > 0:
                    # Calculate average metrics
                    avg_loss = log_loss / log_steps
                    avg_distill_loss = log_distill_loss / log_steps
                    avg_task_loss = log_task_loss / log_steps
                    avg_accuracy = log_accuracy / log_steps
                    
                    # Calculate throughput
                    elapsed = time.time() - start_time
                    throughput = (log_steps * self.batch_size) / elapsed
                    
                    # Update progress bar
                    epoch_iterator.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{avg_accuracy:.4f}",
                        "lr": f"{metrics['lr']:.2e}",
                    })
                    
                    # Log to W&B
                    if self.use_wandb:
                        wandb.log({
                            "loss": avg_loss,
                            "distill_loss": avg_distill_loss,
                            "task_loss": avg_task_loss,
                            "accuracy": avg_accuracy,
                            "learning_rate": metrics["lr"],
                            "throughput": throughput,
                            "step": self.global_step,
                            "epoch": epoch,
                        })
                    
                    # Reset metrics
                    log_loss = 0
                    log_distill_loss = 0
                    log_task_loss = 0
                    log_accuracy = 0
                    log_steps = 0
                    start_time = time.time()
                
                # Evaluate
                if (self.global_step + 1) % self.eval_every == 0:
                    print(f"\nEvaluating at step {self.global_step+1}...")
                    eval_metrics = self.evaluate()
                    
                    # Log evaluation metrics
                    if self.use_wandb:
                        wandb.log({
                            **eval_metrics,
                            "step": self.global_step,
                            "epoch": epoch,
                        })
                    
                    # Print metrics
                    print(f"Validation loss: {eval_metrics['val_loss']:.4f}")
                    print(f"Validation accuracy: {eval_metrics['val_accuracy']:.4f}")
                    
                    # Save best model
                    if eval_metrics["val_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["val_loss"]
                        self.save_checkpoint(
                            path=self.output_dir / "best_model",
                            metrics=eval_metrics
                        )
                        print(f"New best model saved with loss {self.best_eval_loss:.4f}")
                
                # Save checkpoint
                if (self.global_step + 1) % self.save_every == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # End of epoch
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(
                path=self.output_dir / f"epoch-{epoch+1}",
                metrics={"epoch": epoch+1}
            )
        
        # Final evaluation
        print("\nFinal evaluation...")
        eval_metrics = self.evaluate()
        
        # Log final metrics
        if self.use_wandb:
            wandb.log({
                **eval_metrics,
                "step": self.global_step,
                "epoch": self.num_epochs,
                "final": True,
            })
        
        # Save final model
        self.save_checkpoint(
            path=self.output_dir / "final_model",
            metrics=eval_metrics
        )
        
        print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein language model distillation")
    
    # Model parameters
    parser.add_argument("--teacher_model", type=str, default="Rostlab/prot_bert",
                        help="HuggingFace model name for teacher")
    parser.add_argument("--d_model", type=int, default=384,
                        help="Hidden dimension of student model")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of attention heads in student model")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of layers in student model")
    
    # Training parameters
    parser.add_argument("--data_path", type=str, default="data/unref_50",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="model_distill/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--mask_probability", type=float, default=0.15,
                        help="Probability of masking tokens for MLM")
    parser.add_argument("--distill_weight", type=float, default=0.5,
                        help="Weight for distillation loss")
    parser.add_argument("--task_weight", type=float, default=0.5,
                        help="Weight for task loss (MLM)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for distillation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization for memory efficiency")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use W&B for logging")
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log metrics every N steps")
    
    args = parser.parse_args()
    
    # Initialize tokenizer to get vocab_size
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model)
    
    # Create student config
    student_config = {
        "vocab_size": len(tokenizer),  # Add vocab_size from tokenizer
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_model * 4,
        "max_seq_len": args.max_seq_len,
        "dropout": 0.1,
    }
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model_name=args.teacher_model,
        student_config=student_config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mask_probability=args.mask_probability,
        distill_weight=args.distill_weight,
        task_weight=args.task_weight,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        grad_accumulation_steps=args.grad_accumulation_steps,
        device=args.device,
        use_wandb=args.use_wandb,
        fp16=args.fp16,
        use_8bit=args.use_8bit,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )
    
    # Train
    trainer.train() 