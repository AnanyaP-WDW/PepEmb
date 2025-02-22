"""Module for collecting and processing peptide sequence data from various sources."""

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

class PeptideDataCollector:
    """Class to collect peptide data from different sources.
    
    Currently supports:
    - UniProt (~5000 sequences)
    - Random generation (~1000 sequences)
    """

    def __init__(self, cache_dir="peptide_data"):
        """Initialize collector with cache directory.
        
        Args:
            cache_dir (str): Directory path to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_uniprot_peptides(self, max_entries=5000, min_seq_len=2, max_seq_len=60):
        """Download peptide sequences from UniProt.
        
        Args:
            max_entries (int): Maximum number of sequences to return
            min_seq_len (int): Minimum sequence length
            max_seq_len (int): Maximum sequence length
            
        Returns:
            pd.DataFrame: DataFrame containing peptide sequences
        """
        
        base_url = "https://rest.uniprot.org/uniprotkb/search?format=tsv"
        base_url += "&fields=sequence,length,organism_name"
        base_url += f"&query=length:[{min_seq_len} TO {max_seq_len}]"
        
        all_data = []
        page_size = 500  # UniProt's maximum page size
        offset = 0
        
        print("Downloading from UniProt...")
        while offset < max_entries:
            current_size = min(page_size, max_entries - offset)
            url = f"{base_url}&size={current_size}&offset={offset}"
            
            response = requests.get(url)
            df = pd.read_csv(StringIO(response.text), sep='\t')
            
            if df.empty:
                break
                
            all_data.append(df)
            offset += page_size
            print(f"Downloaded {len(all_data) * page_size} sequences...")
            
        if not all_data:
            return pd.DataFrame()
            
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.columns = final_df.columns.str.lower()  # Normalize column names
        return final_df.head(max_entries)

    def generate_random_peptides(self, num_peptides=1000, min_seq_len=2, max_seq_len=60):
        """Generate random peptide sequences.
        
        Args:
            num_peptides (int): Number of sequences to generate
            min_seq_len (int): Minimum sequence length
            max_seq_len (int): Maximum sequence length
            
        Returns:
            pd.DataFrame: DataFrame with sequence, length and organism columns
        """
        import random

        # Standard amino acids only (20)
        # Excluding rare/modified amino acids
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        peptides = []
        peptide_lens = []

        for _ in range(num_peptides):
            length = random.randint(min_seq_len, max_seq_len)
            sequence = ''.join(random.choices(amino_acids, k=length))
            peptides.append(sequence)
            peptide_lens.append(length)

        return pd.DataFrame({
            'sequence': peptides,
            'length': peptide_lens,
            'organism': ['random'] * num_peptides
        })


def filter_valid_peptides(df, sequence_column='sequence', min_length=2, max_length=60):
    """Filter peptide sequences based on length and valid amino acids.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sequence_column (str): Name of sequence column
        min_length (int): Minimum sequence length
        max_length (int): Maximum sequence length
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    def is_valid_sequence(seq):
        return (
            isinstance(seq, str) and
            min_length <= len(seq) <= max_length and
            all(aa in valid_aas for aa in seq)
        )

    return df[df[sequence_column].apply(is_valid_sequence)]


def combine_datasets(collector):
    """Combine peptides from multiple sources and remove duplicates.
    
    Args:
        collector (PeptideDataCollector): Data collector instance
        
    Returns:
        pd.DataFrame: Combined and deduplicated DataFrame
    """
    
    datasets = {
        'uniprot': collector.get_uniprot_peptides(),
        'random': collector.generate_random_peptides()
    }

    all_peptides = []
    for source, df in datasets.items():
        if 'sequence' not in df.columns:
            print(f"Warning: No sequence column in {source} dataset")
            continue

        df = filter_valid_peptides(df)
        df['source'] = source
        all_peptides.append(df)

    combined_df = pd.concat(all_peptides, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['sequence'])

    return combined_df


if __name__ == "__main__":
    collector = PeptideDataCollector()
    print("Collecting peptides from multiple sources...")
    combined_df = combine_datasets(collector)

    print("\nDataset summary:")
    print(f"Total unique peptides: {len(combined_df)}")
    print("\nPeptides by source:")
    print(combined_df['source'].value_counts())

    output_file = f"{collector.cache_dir}/combined_peptides.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined dataset to {output_file}")

    print("\nSample peptides:")
    print(combined_df[['sequence', 'source']].head())