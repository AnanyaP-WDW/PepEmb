#!/usr/bin/env python
import os
import sys
import argparse
import gzip
import requests
import random
from tqdm import tqdm
from pathlib import Path

def download_chunk(url, start_byte, end_byte, chunk_size=8192):
    """Download a specific byte range from a URL"""
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code not in [200, 206]:
        raise Exception(f"Failed to download: HTTP {response.status_code}")
    
    content = b''
    total_size = int(response.headers.get('content-length', 0))
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                content += chunk
                pbar.update(len(chunk))
    
    return content

def parse_fasta_from_bytes(content, max_sequences=1000, min_length=10, max_length=1024):
    """Parse FASTA sequences from downloaded bytes"""
    text = content.decode('utf-8', errors='ignore')
    lines = text.split('\n')
    
    sequences = []
    current_header = None
    current_sequence = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_header and current_sequence:
                if min_length <= len(current_sequence) <= max_length:
                    sequences.append((current_header, current_sequence))
            
            # Start new sequence
            current_header = line
            current_sequence = ""
        else:
            current_sequence += line
    
    # Add the last sequence
    if current_header and current_sequence:
        if min_length <= len(current_sequence) <= max_length:
            sequences.append((current_header, current_sequence))
    
    # Limit to max_sequences
    if len(sequences) > max_sequences:
        sequences = random.sample(sequences, max_sequences)
    
    return sequences

def download_sample_uniref50(output_dir, num_sequences=1000):
    """Download a sample of UniRef50 sequences"""
    uniref50_url = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    output_dir = Path(output_dir)
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # First, download a small chunk to get an idea of the file structure
    # Most protein sequences are < 1000 characters, so 5MB should give us plenty
    print(f"Downloading sample from UniRef50...")
    content = download_chunk(uniref50_url, 0, 5 * 1024 * 1024)
    
    # Decompress the gzipped content
    try:
        decompressed = gzip.decompress(content)
    except Exception as e:
        print(f"Error decompressing content: {e}")
        print("Trying to parse without decompression...")
        decompressed = content
    
    # Parse sequences
    print(f"Parsing sequences...")
    sequences = parse_fasta_from_bytes(decompressed, max_sequences=num_sequences)
    
    print(f"Found {len(sequences)} sequences")
    
    # Split into train/val/test (80/10/10)
    random.shuffle(sequences)
    train_split = int(0.8 * len(sequences))
    val_split = int(0.9 * len(sequences))
    
    train_sequences = sequences[:train_split]
    val_sequences = sequences[train_split:val_split]
    test_sequences = sequences[val_split:]
    
    # Save sequences to files
    print(f"Saving {len(train_sequences)} training sequences...")
    with open(train_dir / "sequences.fasta", "w") as f:
        for header, sequence in train_sequences:
            f.write(f"{header}\n{sequence}\n")
    
    print(f"Saving {len(val_sequences)} validation sequences...")
    with open(val_dir / "sequences.fasta", "w") as f:
        for header, sequence in val_sequences:
            f.write(f"{header}\n{sequence}\n")
    
    print(f"Saving {len(test_sequences)} test sequences...")
    with open(test_dir / "sequences.fasta", "w") as f:
        for header, sequence in test_sequences:
            f.write(f"{header}\n{sequence}\n")
    
    print(f"Dataset created at {output_dir}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a sample of UniRef50 sequences")
    parser.add_argument("--output_dir", type=str, default="data/unref_50_sample",
                        help="Directory to save the sequences")
    parser.add_argument("--num_sequences", type=int, default=1000,
                        help="Number of sequences to download")
    
    args = parser.parse_args()
    
    download_sample_uniref50(args.output_dir, args.num_sequences) 