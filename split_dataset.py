import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset(
    data_dir: str = "peptide_data",
    val_size: float = 0.1,
    random_state: int = 42
) -> None:
    """
    Split the combined dataset into training and validation sets.
    
    Args:
        data_dir (str): Directory containing the dataset
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
    """
    # Read the dataset
    data_path = Path(data_dir) / "combined_peptides_with_features_scaled_standardized.csv"
    df = pd.read_csv(data_path)
    
    # Perform stratified split based on source
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df['source']
    )
    
    # Save splits to CSV files
    train_df.to_csv(Path(data_dir) / "train.csv", index=False)
    val_df.to_csv(Path(data_dir) / "val.csv", index=False)
    
    # Print summary
    print("\nDataset split summary:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({100 * len(train_df)/len(df):.1f}%)")
    print(f"Validation samples: {len(val_df)} ({100 * len(val_df)/len(df):.1f}%)")
    
    print("\nSource distribution in training set:")
    print(train_df['source'].value_counts(normalize=True))
    
    print("\nSource distribution in validation set:")
    print(val_df['source'].value_counts(normalize=True))

if __name__ == "__main__":
    split_dataset()
