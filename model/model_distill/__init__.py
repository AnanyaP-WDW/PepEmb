"""
PepEmb model distillation package.

This package contains code for knowledge distillation from ProtBert 
to a smaller model with RoPE positional embeddings.
"""

# Make sure relative imports work correctly
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core classes
from model_distill.architecture import ProteinTransformerStudent, RoPEPositionalEncoding
from model_distill.dataloader import ProteinDataset, MLMCollator 