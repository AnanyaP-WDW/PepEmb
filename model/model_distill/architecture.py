import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) implementation
    """
    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create rotational frequency matrix
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len=None):
        """
        Apply RoPE to input tensor
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            seq_len: Length of sequence (optional)
        """
        if seq_len is None:
            seq_len = x.size(1)
            
        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sincos_pos = torch.einsum('i,j->ij', seq_idx, self.inv_freq)
        sin, cos = torch.sin(sincos_pos), torch.cos(sincos_pos)
        
        # Reshape for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model//2]
        
        # Apply rotary embeddings
        # Reshape x to separate dimensions for RoPE application
        x_reshape = x.view(*x.shape[:-1], -1, 2)
        
        # Apply rotation
        x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        # Stack and reshape back
        rope_x = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rope_x.flatten(-2)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Store attention weights for visualization/explainability
        self.attention_weights = None
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Ensure mask has the right shape for broadcasting with attention scores
            # scores shape: [batch_size, num_heads, seq_len, seq_len]
            # mask shape should be broadcastable to this
            
            # If mask is [batch_size, seq_len, seq_len], expand to [batch_size, 1, seq_len, seq_len]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
                
            # Apply mask - ensure broadcasting works correctly
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for visualization
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        
        if return_attention:
            return output, attention_weights
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attention=False):
        # Self-attention with residual connection and layer norm
        if return_attention:
            attn_output, attention = self.self_attn(x, x, x, mask, return_attention=True)
            x = self.norm1(x + self.dropout(attn_output))
            # Feed forward with residual connection and layer norm
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
            return x, attention
        else:
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            # Feed forward with residual connection and layer norm
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
            return x

class ProteinTransformerStudent(nn.Module):
    """Student model with RoPE positional embeddings instead of standard positional embeddings"""
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # RoPE positional encoding
        self.pos_encoding = RoPEPositionalEncoding(d_model, max_seq_len)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # MLM prediction head
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input tensor of token ids [batch_size, seq_len]
            mask: Attention mask [batch_size, 1, seq_len, seq_len]
            return_attention: Whether to return attention weights
        """
        # Get input embeddings
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Apply RoPE
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        attentions = []
        # Apply transformer layers
        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, mask, return_attention=True)
                attentions.append(attention)
            else:
                x = layer(x, mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # MLM prediction
        logits = self.mlm_head(x)
        
        if return_attention:
            return logits, attentions
        return logits
    
    def get_embeddings(self, x, mask=None):
        """Get embeddings for downstream tasks"""
        # Get input embeddings
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Apply RoPE
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        return x 