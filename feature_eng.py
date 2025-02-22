'''
This file is used to extract features from the peptide sequences, normalize and scale them.
Instead of using all the features of lets say CTD, this script will create a single new feature.
'''

# feaeture sclaing class that takes in the dataframe as self.cache dir
# this has many menhtod for each column to combine all the values of the dict into one
# this class also has method for standardizing the data, by subtracting each value by the mean and dividing by the standard deviation
# finally it has a method to output the csv in the same directory as the cached_dir

import pandas as pd
import ast


class FeatureScaling:
    '''
    A class to scale and normalize peptide sequence features.
    
    Attributes:
        cached_dir (str): Directory containing the input data files
        df (pd.DataFrame): DataFrame containing the peptide features
    '''
    def __init__(self, cached_dir:str="peptide_data"):
        self.cached_dir = cached_dir
        self.df = pd.read_csv(f"{self.cached_dir}/combined_peptides_with_features.csv")

    
    def convert_ctd_to_score(self, ctd_dict:dict[str, float]) -> float:
        '''
        Convert CTD (Composition, Transition, Distribution) features into a single weighted score.
        
        Args:
            ctd_dict (dict[str, float]): Dictionary containing CTD features
            
        Returns:
            float: Weighted score combining C, T and D values with weights 30%, 20%, 50% respectively
        '''
        # Separate C, T, and D values and weight them differently
        c_values = {k: v for k, v in ctd_dict.items() if k.endswith('C1') or k.endswith('C2') or k.endswith('C3')}
        t_values = {k: v for k, v in ctd_dict.items() if 'T' in k}
        d_values = {k: v for k, v in ctd_dict.items() if 'D' in k}
        
        # Weight: Distribution (50%), Composition (30%), Transition (20%)
        score = (
            0.5 * sum(d_values.values()) / len(d_values) +
            0.3 * sum(c_values.values()) / len(c_values) +
            0.2 * sum(t_values.values()) / len(t_values)
        )
        
        return score
    
    def convert_charge_to_score(self, charge_dict:dict[str, float]) -> float:
        '''
        Convert charge features into a single score by summing all values.
        
        Args:
            charge_dict (dict[str, float]): Dictionary containing charge features
            
        Returns:
            float: Sum of all charge values
        '''
        return sum(charge_dict.values())
    
    def convert_hydrophobicity_to_score(self, hydrophobicity_dict:dict[str, float]) -> float:
        '''
        Convert hydrophobicity features into a single score by summing all values.
        
        Args:
            hydrophobicity_dict (dict[str, float]): Dictionary containing hydrophobicity features
            
        Returns:
            float: Sum of all hydrophobicity values
        '''
        return sum(hydrophobicity_dict.values())
    
    def convert_polarity_to_score(self, polarity_dict:dict[str, float]) -> float:
        '''
        Convert polarity features into a single score by summing all values.
        
        Args:
            polarity_dict (dict[str, float]): Dictionary containing polarity features
            
        Returns:
            float: Sum of all polarity values
        '''
        return sum(polarity_dict.values())
    
    def populate_new_features(self) -> pd.DataFrame:
        '''
        Create new scaled features from the raw feature dictionaries.
        
        Returns:
            pd.DataFrame: DataFrame with new scaled features added:
                         CTD_scaled, Charge_scaled, Hydrophobicity_scaled, Polarity_scaled
        '''
        self.df['CTD_scaled'] = self.df.apply(lambda row: self.convert_ctd_to_score(ast.literal_eval(row['CTD'])), axis=1)
        self.df['Charge_scaled'] = self.df.apply(lambda row: self.convert_charge_to_score(ast.literal_eval(row['Charge'])), axis=1)
        self.df['Hydrophobicity_scaled'] = self.df.apply(lambda row: self.convert_hydrophobicity_to_score(ast.literal_eval(row['Hydrophobicity'])), axis=1)
        self.df['Polarity_scaled'] = self.df.apply(lambda row: self.convert_polarity_to_score(ast.literal_eval(row['Polarity'])), axis=1)

        return self.df

    def standardize_data(self) -> pd.DataFrame:
        '''
        Standardize all numeric columns in the dataset using z-score normalization.
        
        Returns:
            pd.DataFrame: DataFrame with all numeric columns standardized 
                         ((value - mean) / std_dev)
        '''
        df_scaled = self.populate_new_features()
        df_scaled_standardized = df_scaled.copy()
        
        # Include all numeric types: int8/16/32/64, uint8/16/32/64, float16/32/64
        numeric_columns = df_scaled_standardized.select_dtypes(
            include=['int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64',
                    'float16', 'float32', 'float64']
        ).columns
        
        for column in numeric_columns:
            df_scaled_standardized[column] = (
                df_scaled_standardized[column] - df_scaled_standardized[column].mean()
            ) / df_scaled_standardized[column].std()
        
        return df_scaled_standardized
    
    def save_to_csv(self) -> None:
        '''
        Save the standardized dataset to a CSV file.
        
        The file will be saved in the cached_dir with name:
        'combined_peptides_with_features_scaled_standardized.csv'
        '''
        df_scaled_standardized = self.standardize_data()
        df_scaled_standardized.to_csv(f"{self.cached_dir}/combined_peptides_with_features_scaled_standardized.csv", index=False)
        return None
       
if __name__ == "__main__":
    feature_scaling = FeatureScaling()
    feature_scaling.save_to_csv()


