import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Gating mechanism
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))  # Note the ~ to invert the mask

        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Compute gate values
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Apply gating and output projection
        output = self.out_proj(attn_output * gate)
        
        return output

class CustomBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # First sub-block: Gated Self-Attention + LayerNorm
        self.gated_attention = GatedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Second sub-block: Feed-Forward + LayerNorm
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # First sub-block
        #print("pre gated attention")
        attn_output = self.gated_attention(x, mask)
        #print("post gated attention")
        x = self.norm1(x + attn_output)  # Residual connection
        #print("post norm1")
        
        # Second sub-block
        #print("pre ff layer")
        ff_output = self.ff_layer(x)
        #print("post ff layer")
        x = self.norm2(x + ff_output)  # Residual connection
        #print("post norm2")
        
        return x

class CustomPeptideEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 60,
        num_features: int = 4
    ):
        super().__init__()
        
        #dmodel required to be divisible by num_heads and used by mlm head  
        self.d_model = d_model
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # Embeddings
        self.amino_embedding = nn.Embedding(vocab_size, d_model)
        self.feature_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Stack of custom blocks
        self.blocks = nn.ModuleList([
            CustomBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        amino_seq: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Get sequence embeddings
        #print("pre amino acid embedding")
        seq_embed = self.amino_embedding(amino_seq)
        #print("post amino acid embedding")
        # Add positional encodings
        #print("pre pos encoder")
        seq_embed = seq_embed + self.pos_encoder[:, :seq_embed.size(1)]
        #print("post pos encoder")
        #print("pre feature projection")
        # Project features and properly expand to sequence length
        feat_embed = self.feature_projection(features)
        #print("post feature projection")
        feat_embed = feat_embed.unsqueeze(1)
        #print("pre expand")
        feat_embed = feat_embed.expand(-1, amino_seq.size(1), -1)
        #print("post expand")
        
        # Now the dimensions match for addition
        #print("pre add")
        x = seq_embed + feat_embed
        #print("post add")
        #print("pre blocks")
        # Process through blocks
        for block in self.blocks:
            x = block(x, mask)
        #print("post blocks")
        #print("pre final norm")
        return self.final_norm(x)

class MLMPeptideModel(nn.Module):
    def __init__(self, encoder: CustomPeptideEncoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model
        
        # MLM head with weight tying (improves stability)
        self.mlm_head = nn.Linear(self.d_model, vocab_size)
        if hasattr(encoder, 'amino_embedding'):
            self.mlm_head.weight = encoder.amino_embedding.weight  # Weight tying
        
        # LayerNorm for MLM head (improves training stability)
        self.mlm_layer_norm = nn.LayerNorm(self.d_model)
        
        # Initialize MLM head properly
        self._init_mlm_head()

    def _init_mlm_head(self):
        nn.init.normal_(self.mlm_head.weight, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)

    def forward(
        self,
        amino_seq: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        encoded = self.encoder(amino_seq, features, mask)
        encoded = self.mlm_layer_norm(encoded)
        return self.mlm_head(encoded)

if __name__ == "__main__":
    # Update test code for MLM model
    vocab_size = 25
    encoder = CustomPeptideEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_seq_length=60,
        num_features=4
    )
    model = MLMPeptideModel(encoder, vocab_size)
    
    # Dummy inputs
    amino_seq = torch.randint(0, vocab_size, (8, 60))
    features = torch.randn(8, 4)
    mask = torch.ones(8, 60).bool()
    
    # Forward pass
    logits = model(amino_seq, features, mask)
    assert logits.shape == (8, 60, vocab_size)
    print(logits.shape)