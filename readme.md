# Training a Peptide Encoder

A transformer-based masked language model for learning amino acid sequence representations. The model uses self-attention mechanisms with custom gating and incorporates protein features for enhanced sequence understanding. Trained using BERT-style masking on peptide sequences to learn contextual amino acid embeddings.

## Features
- Custom encoder architecture with gated self-attention
- Masked language modeling (MLM) pre-training 
- Amino acid sequence embedding
- Protein feature integration
- BERT-style masking strategy (80/10/10)

## Important Note
The commands and code examples below may not be up to date. Please refer to the actual code files.

## Getting Started

### run
in root:

```bash
python create_dataset.py
``` 
this will create a folder named "peptide_data" , inside it it will create "combined_peptides.csv" dataset. This is collected from unpiprot and random generation
columns: sequence, length, organims and source


steps:
1) create dataset
2) add physical and chemcial fearures to each peptide sequence
3) clean, scale (physical and chemical features are a dict of many properties; logic to convert into single floats) and standardize the features
4) build tonkeinzer
5) vocab_size ?
6) core architecture of the encoder
7) dataloader
8) training/ val with masking
7) finetuning with binary classifier LM head for toxicity predivtion of peptide

model architecture:
some basic research that i have done:
Here's a step-by-step guide on how to use these concatenated vectors to train a transformer model using masked language learning (MLM) for peptides:

Step 1: Preparing the Data for MLM

    Sequence Masking: Randomly mask a percentage of amino acids in each sequence (typically 15% in BERT, but you might adjust for peptides):
        80% of the time, replace with [MASK] token.
        10% of the time, replace with a random amino acid.
        10% of the time, keep the original amino acid unchanged (to mimic real-world noise).
    Input Preparation: For each peptide sequence:
        Convert each amino acid to its token index.
        Apply masking as described.
        Prepare labels where the label for each position is the original amino acid if it was masked or changed, or -1 (or any special token for "no prediction needed") if unchanged.


Step 2: Embeddings and Input Processing

    Embedding Creation:
        Use token embeddings for the amino acids, feature embeddings for biochemical properties, and positional embeddings as described previously.
        Concatenate these embeddings for each position in the sequence to form the input to the transformer.
    Input to Transformer:
        For a sequence of length L, you'll have an input tensor of shape (L, combined_embedding_dim) where combined_embedding_dim is the sum of dimensions from all types of embeddings.


Step 3: Transformer Model Setup

    Model Architecture:
        Define your transformer model. Here's a simplified pseudo-code for setting up:

        python

        import torch.nn as nn

        class PeptideTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_encoder_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)  # Token embedding
                self.feature_embedding = nn.Linear(num_properties, d_model)  # Feature embedding
                self.pos_embedding = nn.Embedding(max_seq_length, d_model)  # Positional embedding

                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
                self.output_layer = nn.Linear(d_model, vocab_size)

            def forward(self, token_ids, features, positions):
                # Combine embeddings
                token_embed = self.embedding(token_ids)
                feature_embed = self.feature_embedding(features)
                pos_embed = self.pos_embedding(positions)
                x = token_embed + feature_embed + pos_embed

                # Pass through transformer encoder
                x = self.transformer_encoder(x)

                # Final linear layer to predict amino acids
                return self.output_layer(x)


Step 4: Training Loop

    Batching: Prepare batches of peptide sequences, where each batch includes masked sequences, the original labels for the masked positions, and any necessary metadata like features or positions.
    Forward Pass:
        Feed the concatenated embeddings into the transformer encoder.
        The output of the transformer encodes contextual information which is then passed through an output layer for prediction.
    Loss Calculation:
        Compute the loss only on masked positions. Use Cross-Entropy Loss:
        python

        import torch.nn.functional as F

        def masked_loss(logits, labels):
            mask = (labels != -1)  # -1 or whatever you use for "no prediction needed"
            loss = F.cross_entropy(logits[mask], labels[mask])
            return loss

    Backward Pass and Optimization:
        Backpropagate the loss to adjust all learnable parameters, including embeddings.
        Use an optimizer like Adam for updates:
        python

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss.backward()
        optimizer.step()


Step 5: Model Evaluation

    Validation: On a separate dataset, perform the same masking strategy but use the model to predict masked tokens. Evaluate using accuracy or other relevant metrics.
    Hyperparameter Tuning: Adjust model size (number of layers, attention heads), learning rate, masking strategy, etc., based on validation performance.


Step 6: Iterative Refinement

    Monitor Training: Watch for overfitting by comparing training and validation losses/accuracies.
    Save Checkpoints: Regularly save model states to revert if needed or for further fine-tuning.
    Experimentation: Try different embedding strategies, attention mechanisms, or even different transformer architectures if performance plateaus.


Step 7: Deployment

    Once satisfied with training, save the final model. For deployment, consider model size, inference speed, and how to handle new sequences with unknown properties or lengths.


Remember, the effectiveness of this model will heavily depend on the quality and quantity of your data, how well you've captured the biochemical nuances in your embeddings, and the specific task you're optimizing for. Each step might require fine-tuning based on empirical results and domain knowledge.


Here's a more detailed implementation for training a transformer model on peptide sequences using masked language learning:

Step 1: Imports and Constants
python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Constants
VOCAB_SIZE = 21  # 20 amino acids + 1 special token for [MASK]
D_MODEL = 256  # Dimension of the model
NHEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 6
MAX_SEQ_LEN = 100  # Maximum sequence length
NUM_PROPERTIES = 3  # e.g., charge, hydrophobicity, size
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
MASK_PROB = 0.15  # Probability of masking


Step 2: Dataset and DataLoader
python

class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, features):
        self.sequences = sequences  # List of tokenized sequences
        self.labels = labels  # Labels for masked positions
        self.features = features  # Biochemical features for each amino acid

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'features': self.features[idx],
            'label': self.labels[idx]
        }

def collate_fn(batch):
    sequences = [torch.tensor(item['sequence']) for item in batch]
    features = [torch.tensor(item['features']) for item in batch]
    labels = [torch.tensor(item['label']) for item in batch]

    # Pad sequences to the same length
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    features = pad_sequence(features, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # -1 for no prediction

    return {
        'sequence': sequences,
        'features': features,
        'label': labels
    }

# Assuming you have these lists prepared
sequences = [...]  # Tokenized sequences where masked tokens are replaced by their index
labels = [...]     # Original tokens for masked positions, -1 elsewhere
features = [...]   # 3D array where each sequence has features for each amino acid

dataset = PeptideDataset(sequences, labels, features)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)


Step 3: Model Architecture
python

class PeptideTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_properties, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.feature_embedding = nn.Linear(num_properties, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, features, positions):
        # Token embeddings
        token_embed = self.embedding(token_ids)
        # Feature embeddings
        feature_embed = self.feature_embedding(features)
        # Positional embeddings
        pos_embed = self.pos_embedding(positions)

        # Combine embeddings
        x = token_embed + feature_embed + pos_embed

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Output layer
        return self.output_layer(x)

model = PeptideTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_PROPERTIES, MAX_SEQ_LEN)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


Step 4: Training Loop
python

def mask_sequence(seq, mask_prob, vocab_size):
    mask_token = VOCAB_SIZE - 1  # Assuming [MASK] is the last token in vocab
    masked_seq = seq.clone()
    labels = torch.full_like(seq, -1)  # No prediction needed by default
    
    mask = torch.rand(seq.shape) < mask_prob
    masked_seq[mask] = mask_token
    labels[mask] = seq[mask]
    
    return masked_seq, labels

def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Masking
            masked_seq, labels = mask_sequence(batch['sequence'], MASK_PROB, VOCAB_SIZE)
            
            # Prepare positions
            positions = torch.arange(MAX_SEQ_LEN).expand(batch['sequence'].shape[0], MAX_SEQ_LEN).to(batch['sequence'].device)
            
            # Forward pass
            logits = model(masked_seq, batch['features'], positions)
            
            # Compute loss only on masked positions
            mask = (labels != -1)
            loss = F.cross_entropy(logits[mask], labels[mask])
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

train(model, dataloader, optimizer, EPOCHS)


Notes:

    Masking: Here, we're doing a simple version of masking. In practice, you might want to implement the exact BERT masking strategy (80% [MASK], 10% random, 10% keep original).
    Features: Ensure your feature data is normalized or scaled appropriately before feeding into the model.
    Positional Encoding: This example uses learned positional encodings, but you could also use fixed sine-cosine encodings.
    Evaluation: You'd want to include validation steps to check for overfitting and to save the best model.


This code snippet provides a foundation for training an encoder-only transformer model for peptide sequences using masked language learning. Remember to adjust parameters, handle edge cases, and use proper data splitting for validation and testing.


### References:
1) Peptide transofomer with amsking - https://academic.oup.com/bib/article/24/6/bbad399/7434461
code - https://github.com/horsepurve/DeepB3P3/blob/main/DeepB3P3.py
2) PeptideBert - https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398
3) ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing - (https://arxiv.org/abs/2007.06225)