# Protein Language Model Distillation

This directory contains code for knowledge distillation from ProtBert (teacher model) to a smaller, more efficient protein language model (student model) with RoPE positional embeddings.

## Features

- **Student model architecture**: Transformer-based architecture with Rotary Position Embeddings (RoPE) instead of standard positional embeddings
- **Knowledge distillation**: Distill knowledge from the larger ProtBert model to the smaller student model
- **Masked Language Modeling**: Train on MLM task with 15% masking rate (like BERT)
- **Multi-device support**: Run on CPU or GPU with mixed precision training
- **Training monitoring**: Comprehensive training metrics with optional W&B integration
- **Explainability**: Generate saliency maps and attention visualizations for model interpretability

## Conceptual Diagram

```
                             KNOWLEDGE DISTILLATION WORKFLOW
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌─────────────────┐             ┌────────────────────────────────────────┐  │
│  │                 │             │                                        │  │
│  │  Protein Data   │             │            Data Processing             │  │
│  │  (UniRef50)     │────────────▶│  - Tokenization                        │  │
│  │                 │             │  - Masking (15% of tokens)             │  │
│  │                 │             │  - Train/Val/Test Split                │  │
│  │                 │             │                                        │  │
│  │                 │             └──────────────────┬─────────────────────┘  │
│  │                                                     │                        │
│  │                                                     │                        │
│  │  ┌──────────────────┐     ┌─────────────────┐     ┌────────────────┐  │  │
│  │  │                  │     │                 │     │                │  │  │
│  │  │  Teacher Model   │     │  Masked Input   │     │ Target Labels  │  │  │
│  │  │  (ProtBert)      │     │                 │     │                │  │  │
│  │  │  [Frozen]        │     │                 │     │                │  │  │
│  │  └─────────┬────────┘     └────────┬────────┘     └────────┬───────┘  │  │
│  │            │                       │                       │          │  │
│  │            ▼                       │                       │          │  │
│  │  ┌─────────────────┐              │                       │          │  │
│  │  │                 │              │                       │          │  │
│  │  │ Teacher Logits  │              │                       │          │  │
│  │  │                 │              │                       │          │  │
│  │  └─────────┬───────┘              │                       │          │  │
│  │            │                      ▼                       │          │  │
│  │            │          ┌────────────────────┐              │          │  │
│  │            │          │                    │              │          │  │
│  │            │          │   Student Model    │              │          │  │
│  │            │          │   (RoPE Embeddings)│              │          │  │
│  │            │          │                    │              │          │  │
│  │            │          └─────────┬──────────┘              │          │  │
│  │            │                    │                         │          │  │
│  │            │                    ▼                         │          │  │
│  │            │          ┌─────────────────┐                 │          │  │
│  │            └─────────▶│                 │◀────────────────┘          │  │
│  │                       │  Loss Functions │                            │  │
│  │                       │  - KL Divergence│                            │  │
│  │                       │  - Cross Entropy│                            │  │
│  │                       │                 │                            │  │
│  │                       └────────┬────────┘                            │  │
│  │                                │                                     │  │
│  │                                ▼                                     │  │
│  │                      ┌──────────────────┐                            │  │
│  │                      │                  │                            │  │
│  │                      │    Optimizer     │                            │  │
│  │                      │                  │                            │  │
│  │                      └──────────────────┘                            │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │                       MONITORING & VISUALIZATION                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐    │  │
│  │  │                 │  │                 │  │                     │    │  │
│  │  │  W&B Tracking   │  │ Saliency Maps   │  │  Attention Patterns │    │  │
│  │  │  - Loss         │  │ - Integrated    │  │  - Layer-by-layer   │    │  │
│  │  │  - Accuracy     │  │   Gradients     │  │  - Head-by-head     │    │  │
│  │  │  - Throughput   │  │                 │  │                     │    │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘    │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


                       STUDENT MODEL ARCHITECTURE
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌────────────────────────┐                                      │
│  │                        │                                      │
│  │    Input Tokens        │                                      │
│  │                        │                                      │
│  └───────────┬────────────┘                                      │
│              │                                                   │
│              ▼                                                   │
│  ┌────────────────────────┐                                      │
│  │                        │                                      │
│  │   Token Embeddings     │                                      │
│  │                        │                                      │
│  └───────────┬────────────┘                                      │
│              │                                                   │
│              ▼                                                   │
│  ┌────────────────────────┐                                      │
│  │                        │                 ┌──────────────────┐ │
│  │     RoPE Positional    │                 │                  │ │
│  │      Embeddings        │                 │  Rotational      │ │
│  │                        │◀────────────────┤  Frequencies     │ │
│  └───────────┬────────────┘                 │                  │ │
│              │                              └──────────────────┘ │
│              ▼                                                   │
│  ┌────────────────────────┐                                      │
│  │     Transformer        │                                      │
│  │       Layers           │                                      │
│  │  ┌──────────────────┐  │                                      │
│  │  │  Self-Attention  │  │                                      │
│  │  │  ┌────────────┐  │  │                                      │
│  │  │  │ Multi-Head │  │  │                                      │
│  │  │  │ Attention  │  │  │                                      │
│  │  │  └────────────┘  │  │                                      │
│  │  └──────────────────┘  │                                      │
│  │  ┌──────────────────┐  │                                      │
│  │  │   Feed Forward   │  │                                      │
│  │  │     Network      │  │                                      │
│  │  └──────────────────┘  │                                      │
│  │  ┌──────────────────┐  │                                      │
│  │  │   Layer Norm     │  │                                      │
│  │  └──────────────────┘  │                                      │
│  └───────────┬────────────┘                                      │
│              │                                                   │
│              ▼                                                   │
│  ┌────────────────────────┐                                      │
│  │                        │                                      │
│  │    Output Layer        │                                      │
│  │  (MLM Prediction Head) │                                      │
│  │                        │                                      │
│  └────────────────────────┘                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
model_distill/
├── architecture.py          # Student model architecture with RoPE embeddings
├── distill.py               # Knowledge distillation training script
├── visualization.py         # Explainability and visualization utilities
├── download_sample_data.py  # Script to download a small subset of UniRef50
├── setup_wandb.py           # Script to set up Weights & Biases monitoring
├── run_test_distillation.sh # Shell script to run the entire pipeline
└── README.md                # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.15+
- Captum (for saliency maps)
- Matplotlib and Seaborn (for visualizations)
- Wandb (optional, for logging)

## Quick Start with Sample Data

For a quick test run with a small subset of UniRef50 (1000 sequences), run:

```bash
# Make the script executable
chmod +x model/model_distill/run_test_distillation.sh

# Run the test distillation pipeline
./model/model_distill/run_test_distillation.sh
```

This script:
1. Downloads a small sample of sequences from UniRef50
2. (Optional) Sets up Weights & Biases for monitoring
3. Runs distillation with a smaller model configuration
4. Generates visualizations for a sample sequence

## Setting Up Weights & Biases Monitoring

To use the free version of Weights & Biases for monitoring:

1. Set up W&B with:
   ```bash
   python model/model_distill/setup_wandb.py
   ```

2. Follow the prompts to create a free account or log in with your API key

3. Run a test to verify W&B is working:
   ```bash
   python model/model_distill/setup_wandb.py --test
   ```

4. Add the `--use_wandb` flag to your training command:
   ```bash
   python model_distill/distill.py --use_wandb [other args...]
   ```

## Downloading Small Sample Data

To download just a small subset of UniRef50 (e.g., 1000 sequences):

```bash
python model/model_distill/download_sample_data.py \
  --output_dir data/unref_50_sample \
  --num_sequences 1000
```

This will create a small dataset with train/val/test splits for testing purposes.

## Full Training with Knowledge Distillation

For full training with the complete UniRef50 dataset:

```bash
python model_distill/distill.py \
  --data_path data/unref_50 \
  --output_dir model_distill/checkpoints \
  --teacher_model Rostlab/prot_bert \
  --d_model 384 \
  --num_heads 6 \
  --num_layers 4 \
  --distill_weight 0.5 \
  --task_weight 0.5 \
  --batch_size 32 \
  --max_seq_len 512 \
  --learning_rate 5e-5 \
  --num_epochs 10 \
  --fp16 \
  --use_wandb
```

## Visualization and Explainability

```bash
python model_distill/visualization.py \
  --model_path model_distill/checkpoints/best_model/model.pt \
  --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
  --output_dir visualizations
```

## Training Process

The distillation process involves:

1. **Masking tokens**: 15% of tokens are masked as in BERT
2. **Teacher model prediction**: Pass the input through ProtBert to get teacher logits
3. **Student model prediction**: Pass the same input through the student model
4. **Loss computation**:
   - **Distillation loss**: KL divergence between teacher and student logits
   - **Task loss**: Cross-entropy for masked language modeling
   - **Combined loss**: Weighted sum (e.g., 0.5 each) of distillation and task losses
5. **Optimization**: Update student model parameters to minimize the combined loss

## Best Practices for CPU and GPU Training

### For GPU:

- Enable mixed precision training with `--fp16` flag
- For memory-constrained environments, use 8-bit quantization with `--use_8bit` flag
- Use gradient accumulation for larger effective batch sizes: `--grad_accumulation_steps 4`
- Adjust batch size based on your GPU memory

### For CPU:

- Disable mixed precision (omit `--fp16`)
- Use smaller batch sizes
- Reduce model size (fewer layers, smaller hidden dimensions)
- Consider using multiple CPU threads through DataLoader workers

## Monitoring Training

- **W&B integration**: Enable with `--use_wandb` for comprehensive training visualizations
- **Regular evaluation**: Validating every N steps (configurable with `--eval_every`)
- **Checkpointing**: Save model state at regular intervals and best performing models
- **Metrics tracked**: Loss (distillation, task, combined), accuracy, learning rate, throughput

## Explainability Features

The visualization module provides several explainability features:

1. **Attention visualization**: Visualize attention patterns across all layers and heads
2. **Saliency maps**: Identify which tokens contribute most to predictions
3. **Important token identification**: Automatically identify the most important tokens in a sequence
4. **Layer-specific analysis**: Analyze specific layers or attention heads

## Customization

You can customize various aspects of the distillation process:

- **Model architecture**: Adjust the number of layers, hidden dimensions, and attention heads
- **Loss weighting**: Change the balance between distillation and task losses
- **Temperature**: Adjust the softness of the teacher's probability distribution
- **Training schedule**: Customize learning rate, warmup steps, and training duration

## Citation

If you use this code, please cite:

```
@misc{PepEmb,
  author = {Your Name},
  title = {PepEmb: Efficient Protein Language Models with Knowledge Distillation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/PepEmb}}
}
``` 