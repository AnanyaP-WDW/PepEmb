# Populate Dataset
# open csv "combined_peptides.csv" from "peptide_data" folder

# from PyBioMed.PyProtein import AAComposition, DipeptideComposition, Autocorrelation, CTD, GetProDes
# import pandas as pd

# def extract_protein_features(sequence):
#     features = {
#         'Sequence': [sequence],
#         'AAC': AAComposition.CalculateAAComposition(sequence),
#         'Dipeptide': DipeptideComposition.CalculateDipeptideComposition(sequence),
#         'CTD': CTD.CalculateCTD(sequence),
#         'Moran': Autocorrelation.CalculateMoranAuto(sequence),
#         'Geary': Autocorrelation.CalculateGearyAuto(sequence),
#         'MoreauBroto': Autocorrelation.CalculateMoreauBrotoAuto(sequence),
#         'PseAAC': GetProDes(sequence)['PseAAC']
#     }
    
#     # Flatten the dictionary for DataFrame creation
#     flat_features = {}
#     for key, value in features.items():
#         if isinstance(value, dict):
#             for sub_key, sub_value in value.items():
#                 flat_features[f"{key}_{sub_key}"] = sub_value
#         else:
#             flat_features[key] = value

#     return pd.DataFrame(flat_features)

# # Example Usage
# peptide_sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAF"

# peptide_data = extract_protein_features(peptide_sequence)
# print(peptide_data)

# modify this above script to populate each sequence in the csv file
# and then create a new csv file with all the new features and new features inside the "peptide_data" folder


import pandas as pd
from PyBioMed.PyProtein import (
    CTD
)

class PopulateDataset:
    '''
    This class is used to populate the dataset with protein features.
    It reads peptide sequences from a CSV file, extracts various protein features,
    and saves the enriched dataset back to a new CSV file.
    
    Attributes:
        cached_dir (str): Directory path where input/output CSV files are stored
        df (pd.DataFrame): DataFrame containing the peptide sequences and features
    '''
    def __init__(self, cached_dir:str="peptide_data"):
        """
        Initialize the PopulateDataset class.
        
        Args:
            cached_dir (str): Directory path for input/output CSV files. Defaults to "peptide_data"
        """
        self.cached_dir = cached_dir
        self.df = pd.read_csv(f"{self.cached_dir}/combined_peptides.csv")
        self.df = self.df.drop_duplicates(subset=['sequence'])
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(columns=['source', 'organism', 'length'])
    
    def extract_protein_features(self, sequence:str) -> dict[str, float]:
        """
        Extract protein features for a given peptide sequence.
        
        Args:
            sequence (str): Peptide sequence to analyze
            
        Returns:
            dict[str, float]: Dictionary containing extracted features including:
                - Sequence: Original peptide sequence
                - CTD: Composition, Transition and Distribution descriptors
                - Charge: Charge-based composition features
                - Hydrophobicity: Hydrophobicity-based composition features
                - Polarity: Polarity-based composition features
        """
        features = {
            'Sequence': sequence,
            'CTD': CTD.CalculateCTD(sequence),
            'Charge' : CTD.CalculateCompositionCharge(sequence),
            'Hydrophobicity' : CTD.CalculateCompositionHydrophobicity(sequence),
            'Polarity' : CTD.CalculateCompositionPolarity(sequence)
        }
        return features

    def extract_features(self) -> pd.DataFrame:
        """
        Extract features for all sequences in the dataset.
        
        Returns:
            pd.DataFrame: DataFrame containing original sequences and their extracted features
        """
        # Apply the feature extraction to each sequence and create a list of features
        features_list = self.df['sequence'].apply(self.extract_protein_features)
        
        # Convert features to DataFrame and concatenate with original
        features_df = pd.DataFrame(features_list.tolist())
        result_df = pd.concat([self.df, features_df], axis=1)
        
        self.df = result_df
        return self.df
    
    def save_to_csv(self, original_df:pd.DataFrame) -> None:
        """
        Save the enriched dataset to a CSV file, merging with original metadata.
        
        Args:
            original_df (pd.DataFrame): Original DataFrame containing metadata like source, 
                                      organism, length to be merged with features
        """
        # the dataframe should be combinedd with self.dir/combined_peptides.csv to also include other features from ther
        # like source, organism, length, etc.
        self.df = pd.merge(self.df, original_df, on='sequence', how='left')

        self.df.to_csv(f"{self.cached_dir}/combined_peptides_with_features.csv", index=False)
        print("Dataset saved to peptide_data/combined_peptides_with_features.csv")

if __name__ == "__main__":
    populate_dataset = PopulateDataset()
    populate_dataset.extract_features()
    populate_dataset.save_to_csv(original_df=pd.read_csv(f"{populate_dataset.cached_dir}/combined_peptides.csv"))


