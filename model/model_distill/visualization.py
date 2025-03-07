import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import BertTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_distill.architecture import ProteinTransformerStudent

class ProteinModelVisualizer:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = "Rostlab/prot_bert",
        device: str = None,
    ):
        """
        Initialize the visualizer with a trained model
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer (or HF model name)
            device: Device to use for inference
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Load config
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            student_config = config.get("student_config", {})
        else:
            # Default config if not found
            student_config = {
                "d_model": 384,
                "num_heads": 6,
                "num_layers": 4,
                "d_ff": 1536,
                "max_seq_len": 512,
                "dropout": 0.1,
            }
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # Update config with vocab size
        student_config["vocab_size"] = len(self.tokenizer)
        
        # Initialize model
        self.model = ProteinTransformerStudent(**student_config)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Set up integrated gradients for saliency
        self.integrated_gradients = LayerIntegratedGradients(
            self.model, self.model.token_embedding
        )
    
    def visualize_attention(
        self,
        sequence: str,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        output_path: Optional[str] = None,
        cmap: str = "viridis",
    ) -> plt.Figure:
        """
        Visualize attention patterns for a protein sequence
        
        Args:
            sequence: Protein sequence to visualize
            layer_idx: Specific layer to visualize (None for all)
            head_idx: Specific attention head to visualize (None for all)
            output_path: Path to save visualization (None to not save)
            cmap: Colormap for visualization
            
        Returns:
            Matplotlib figure with attention visualization
        """
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        # Create attention mask (all 1s since we're handling a single sequence)
        attention_mask = torch.ones_like(input_ids)
        
        # Create attention mask in the correct format
        seq_len = input_ids.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get attention weights for all layers
        with torch.no_grad():
            _, attention_weights = self.model(input_ids, mask=extended_attention_mask, return_attention=True)
        
        # Convert attention weights to numpy
        attention_weights = [attn.cpu().numpy() for attn in attention_weights]
        
        # Determine what to visualize
        if layer_idx is not None and head_idx is not None:
            # Visualize a specific head in a specific layer
            attn_to_viz = attention_weights[layer_idx][:, head_idx, :, :]
            title = f"Layer {layer_idx+1}, Head {head_idx+1}"
            n_rows, n_cols = 1, 1
        elif layer_idx is not None:
            # Visualize all heads in a specific layer
            attn_to_viz = attention_weights[layer_idx][0]  # [n_heads, seq_len, seq_len]
            title = f"Layer {layer_idx+1}, All Heads"
            n_heads = attn_to_viz.shape[0]
            n_rows = int(np.ceil(np.sqrt(n_heads)))
            n_cols = int(np.ceil(n_heads / n_rows))
        elif head_idx is not None:
            # Visualize a specific head across all layers
            attn_to_viz = [attn[0, head_idx, :, :] for attn in attention_weights]
            title = f"Head {head_idx+1}, All Layers"
            n_layers = len(attn_to_viz)
            n_rows = int(np.ceil(np.sqrt(n_layers)))
            n_cols = int(np.ceil(n_layers / n_rows))
        else:
            # Visualize average attention across all heads for each layer
            attn_to_viz = [attn.mean(axis=1)[0] for attn in attention_weights]
            title = "Average Attention Across Heads"
            n_layers = len(attn_to_viz)
            n_rows = int(np.ceil(np.sqrt(n_layers)))
            n_cols = int(np.ceil(n_layers / n_rows))
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
        fig.suptitle(title, fontsize=16)
        
        # Plot attention patterns
        if layer_idx is not None and head_idx is not None:
            # Single attention matrix
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(attn_to_viz[0], cmap=cmap)
            ax.set_title(f"Attention Pattern")
            
            # Add labels if sequence is not too long
            if len(tokens) <= 100:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=8)
            
            plt.colorbar(im, ax=ax)
        
        elif layer_idx is not None:
            # Multiple heads in one layer
            for h in range(attn_to_viz.shape[0]):
                ax = fig.add_subplot(n_rows, n_cols, h + 1)
                im = ax.imshow(attn_to_viz[h], cmap=cmap)
                ax.set_title(f"Head {h+1}")
                
                # Add labels if sequence is not too long
                if len(tokens) <= 50:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=6)
                
                plt.colorbar(im, ax=ax)
        
        elif head_idx is not None or layer_idx is None:
            # Multiple layers (either specific head or average)
            for l, attn in enumerate(attn_to_viz):
                ax = fig.add_subplot(n_rows, n_cols, l + 1)
                im = ax.imshow(attn, cmap=cmap)
                
                if head_idx is not None:
                    ax.set_title(f"Layer {l+1}")
                else:
                    ax.set_title(f"Layer {l+1} (avg)")
                
                # Add labels if sequence is not too long
                if len(tokens) <= 50:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=6)
                
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def generate_saliency_map(
        self,
        sequence: str,
        target_position: Optional[int] = None,
        n_steps: int = 50,
        output_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Generate saliency map for a protein sequence using integrated gradients
        
        Args:
            sequence: Protein sequence to analyze
            target_position: Position to compute saliency for (None for all positions)
            n_steps: Number of steps for integrated gradients
            output_path: Path to save visualization (None to not save)
            
        Returns:
            Tuple of (figure, attributions)
        """
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        # Create attention mask (all 1s since we're handling a single sequence)
        attention_mask = torch.ones_like(input_ids)
        
        # Create attention mask in the correct format for the predict function
        seq_len = input_ids.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Define prediction function for a specific position or sum of all positions
        def predict_fn(inputs):
            logits = self.model(inputs, mask=extended_attention_mask)
            
            if target_position is not None:
                # Return logits for specific position
                return logits[:, target_position, :]
            else:
                # Return sum of logits across sequence
                return logits.sum(dim=1)
        
        # Compute attributions using integrated gradients
        attributions, delta = self.integrated_gradients.attribute(
            input_ids,
            target=None,  # None means the model output is directly used
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        
        # Convert attributions to scalar values per token
        attributions = attributions.sum(dim=-1).cpu().numpy()[0]
        
        # Normalize attributions
        attributions = attributions / np.linalg.norm(attributions)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot saliency bars
        x = np.arange(len(tokens))
        bars = ax.bar(x, attributions, alpha=0.8)
        
        # Color bars based on attribution values
        for i, bar in enumerate(bars):
            if attributions[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Set labels
        if target_position is not None:
            ax.set_title(f"Attribution Scores for Target Position {target_position} ({tokens[target_position]})")
        else:
            ax.set_title("Attribution Scores Across Sequence")
        
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Attribution Score")
        
        # Add sequence labels
        if len(tokens) <= 100:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=90)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, attributions
    
    def analyze_sequence(
        self,
        sequence: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive analysis on a sequence with multiple visualizations
        
        Args:
            sequence: Protein sequence to analyze
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of analysis results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        # Create attention mask in the correct format
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get model outputs
        with torch.no_grad():
            logits, attentions = self.model(input_ids, mask=extended_attention_mask, return_attention=True)
        
        # 1. Generate overall attention visualization (average across heads)
        attention_fig = self.visualize_attention(
            sequence,
            output_path=os.path.join(output_dir, "attention_avg.png") if output_dir else None
        )
        results["attention_fig"] = attention_fig
        
        # 2. Generate layer-by-layer head attention for the last layer
        last_layer_fig = self.visualize_attention(
            sequence,
            layer_idx=len(self.model.layers) - 1,  # Last layer
            output_path=os.path.join(output_dir, "attention_last_layer.png") if output_dir else None
        )
        results["last_layer_fig"] = last_layer_fig
        
        # 3. Generate saliency map for overall sequence
        saliency_fig, attributions = self.generate_saliency_map(
            sequence,
            output_path=os.path.join(output_dir, "saliency_map.png") if output_dir else None
        )
        results["saliency_fig"] = saliency_fig
        results["attributions"] = attributions
        
        # 4. Find the most important tokens based on attributions
        abs_attributions = np.abs(attributions)
        top_indices = np.argsort(abs_attributions)[::-1][:5]  # Top 5 tokens
        
        important_tokens = []
        for idx in top_indices:
            important_tokens.append({
                "position": idx,
                "token": tokens[idx],
                "attribution": attributions[idx]
            })
        
        results["important_tokens"] = important_tokens
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein model visualization")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="Rostlab/prot_bert",
                        help="Path to the tokenizer or HF model name")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Protein sequence to analyze")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to visualize (0-indexed)")
    parser.add_argument("--head", type=int, default=None,
                        help="Specific attention head to visualize (0-indexed)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ProteinModelVisualizer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.layer is not None or args.head is not None:
        # Visualize specific attention weights
        visualizer.visualize_attention(
            sequence=args.sequence,
            layer_idx=args.layer,
            head_idx=args.head,
            output_path=os.path.join(args.output_dir, "attention.png")
        )
        plt.close()
    else:
        # Run comprehensive analysis
        results = visualizer.analyze_sequence(
            sequence=args.sequence,
            output_dir=args.output_dir
        )
        
        # Print important tokens
        print("Most important tokens:")
        for token in results["important_tokens"]:
            print(f"Position {token['position']}: {token['token']} (attribution: {token['attribution']:.4f})") 