#!/bin/bash

# Set error handling
set -e

echo "Starting data preparation pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install pandas requests PyBioMed scikit-learn tqdm

# Step 1: Create initial dataset
echo -e "\n1. Creating initial dataset..."
python create_dataset.py
if [ $? -eq 0 ]; then
    echo "✓ Dataset creation successful"
else
    echo "✗ Dataset creation failed"
    exit 1
fi

# Step 2: Add features to dataset
echo -e "\n2. Adding features to dataset..."
python populate_dataset.py
if [ $? -eq 0 ]; then
    echo "✓ Feature extraction successful"
else
    echo "✗ Feature extraction failed"
    exit 1
fi

# Step 3: Scale and standardize features
echo -e "\n3. Scaling and standardizing features..."
python feature_eng.py
if [ $? -eq 0 ]; then
    echo "✓ Feature engineering successful"
else
    echo "✗ Feature engineering failed"
    exit 1
fi

# Step 4: Split into train/val sets
echo -e "\n4. Splitting dataset into train/val sets..."
python split_dataset.py
if [ $? -eq 0 ]; then
    echo "✓ Dataset splitting successful"
else
    echo "✗ Dataset splitting failed"
    exit 1
fi

echo -e "\nData preparation pipeline completed successfully!"
echo "You can find the processed data in the 'peptide_data' directory"

# Deactivate virtual environment
# deactivate 