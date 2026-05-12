from .tokenizing.prepare_tokenizer_data import prepare_tokenizer_data
from .tokenizing.convert_to_parquet import convert_to_parquet
from .tokenizing.train_tokenizer import train_tokenizer

def create_tokenizer(): 
    """
    Prepares data and trains the tokenizer.
    Run this after downloading and preprocessing the data.
    """

    prepare_tokenizer_data()
    convert_to_parquet()
    train_tokenizer()