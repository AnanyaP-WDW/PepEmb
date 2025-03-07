#!/bin/bash
set -e  # Exit on error

# Define parameters
DATA_DIR="data/unref_50_sample"
OUTPUT_DIR="model_distill/checkpoints/test_run"
NUM_SEQUENCES=1000
BATCH_SIZE=8
MAX_SEQ_LEN=256
NUM_EPOCHS=2
EVAL_EVERY=100
SAVE_EVERY=500
LOG_EVERY=10

# Step 1: Download a sample of data
echo "==== Step 1: Downloading sample data ===="
python model/model_distill/download_sample_data.py \
  --output_dir $DATA_DIR \
  --num_sequences $NUM_SEQUENCES

# Step 2: Setup W&B (optional)
echo "==== Step 2: Setting up Weights & Biases (optional) ===="
echo "If you want to use W&B, run: python model/model_distill/setup_wandb.py"
echo "Then add --use_wandb to the training command"
echo ""

# Step 3: Run distillation
echo "==== Step 3: Running distillation ===="
echo "Starting distillation with smaller student model..."

# Define base command - no spaces after backslashes!
CMD="python model/model_distill/distill.py \
--data_path $DATA_DIR \
--output_dir $OUTPUT_DIR \
--teacher_model Rostlab/prot_bert \
--d_model 256 \
--num_heads 4 \
--num_layers 3 \
--distill_weight 0.5 \
--task_weight 0.5 \
--batch_size $BATCH_SIZE \
--max_seq_len $MAX_SEQ_LEN \
--learning_rate 5e-5 \
--warmup_steps 100 \
--num_epochs $NUM_EPOCHS \
--eval_every $EVAL_EVERY \
--save_every $SAVE_EVERY \
--log_every $LOG_EVERY"

# Uncomment to enable optional features
# CMD="$CMD --use_wandb"  # Enable W&B monitoring
# CMD="$CMD --fp16"       # Enable mixed precision (for GPU)

# Execute command
echo "Running: $CMD"
eval $CMD

# Step 4: Visualize results
echo "==== Step 4: Visualizing results ===="
# Get a sample sequence from the test set
SEQUENCE=$(head -n 2 $DATA_DIR/test/sequences.fasta | tail -n 1 | head -c 50)
echo "Using sequence: $SEQUENCE"

# Generate visualizations
python model/model_distill/visualization.py \
  --model_path $OUTPUT_DIR/best_model/model.pt \
  --sequence "$SEQUENCE" \
  --output_dir "$OUTPUT_DIR/visualizations"

echo "==== All done! ===="
echo "Trained model saved at: $OUTPUT_DIR/best_model"
echo "Visualizations saved at: $OUTPUT_DIR/visualizations" 