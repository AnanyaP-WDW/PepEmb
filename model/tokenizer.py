'''
This file is for tokenizing peptides.
Each amino acid is tokenized into a number.
'''

class Tokenizer:
    def __init__(self):
        """
        Initialize the tokenizer with standard amino acid vocabulary.
        
        The vocabulary includes:
        - 20 standard amino acids (A-Y)
        - Special tokens: PAD (0), UNK (21), BOS (22), EOS (23), MASK (24)
        """
        # Standard amino acid vocabulary
        self.amino_acids = {
            'A': 1,  # Alanine
            'C': 2,  # Cysteine
            'D': 3,  # Aspartic Acid
            'E': 4,  # Glutamic Acid
            'F': 5,  # Phenylalanine
            'G': 6,  # Glycine
            'H': 7,  # Histidine
            'I': 8,  # Isoleucine
            'K': 9,  # Lysine
            'L': 10, # Leucine
            'M': 11, # Methionine
            'N': 12, # Asparagine
            'P': 13, # Proline
            'Q': 14, # Glutamine
            'R': 15, # Arginine
            'S': 16, # Serine
            'T': 17, # Threonine
            'V': 18, # Valine
            'W': 19, # Tryptophan
            'Y': 20, # Tyrosine
            '<PAD>': 0,  # Padding token
            '<UNK>': 21,  # Unknown amino acid token
            '<BOS>': 22, # Begin of sequence
            '<EOS>': 23, # End of sequence
            '<MASK>': 24, # Masking token
        }
        # Create reverse mapping for decoding
        self.id_to_aa = {v: k for k, v in self.amino_acids.items()}

    def tokenize(self, peptide: str) -> list[int]:
        """
        Convert a peptide sequence into a list of integer tokens.
        
        Args:
            peptide (str): Amino acid sequence (e.g., 'ACDEF')
            
        Returns:
            list[int]: List of integer tokens representing the amino acid sequence
        """
        return [self.amino_acids.get(aa, self.amino_acids['<UNK>']) 
                for aa in peptide.upper()]

    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of tokens back to amino acid sequence.
        
        Args:
            tokens (list[int]): List of integer tokens
            
        Returns:
            str: Amino acid sequence reconstructed from tokens
        """
        return ''.join(self.id_to_aa.get(token, '<UNK>') 
                      for token in tokens)

    def pad_sequence(self, tokens: list[int], max_length: int=60) -> list[int]:
        """
        Pad or truncate a token sequence to a fixed length, adding BOS and EOS tokens.
        
        Args:
            tokens (list[int]): List of integer tokens
            max_length (int): Desired sequence length (default: 60)
            
        Returns:
            list[int]: Padded/truncated sequence with BOS and EOS tokens
                      If input length > max_length, sequence is truncated
                      If input length < max_length, sequence is padded with PAD tokens
        """
        # Add BOS and EOS tokens
        tokens = [self.amino_acids['<BOS>']] + tokens + [self.amino_acids['<EOS>']]
        
        # Account for BOS and EOS in max length
        effective_max_length = max_length - 2
        
        if len(tokens) > max_length:
            return tokens[:max_length]
            
        return tokens + [self.amino_acids['<PAD>']] * (max_length - len(tokens))
    
    def get_token_id(self, token: str) -> int:
        """
        Get the token ID for a given token string.
        
        Args:
            token (str): Token string to convert to ID
            
        Returns:
            int: Token ID corresponding to the input token string.
                 Returns UNK token ID if token not found in vocabulary
        """
        return self.amino_acids.get(token, self.amino_acids['<UNK>'])
    
    def get_token_string(self, token_id: int) -> str:
        """
        Get the token string for a given token ID.
        
        Args:
            token_id (int): Token ID to convert to string
            
        Returns:
            str: Token string corresponding to the input token ID.
                 Returns '<UNK>' if token ID not found in vocabulary
        """
        return self.id_to_aa.get(token_id, '<UNK>')
