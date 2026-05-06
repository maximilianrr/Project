
from . import tokenizing

def create_tokenizer(): 
    """
    Prepares data and trains the tokenizer.
    Run this after downloading and preprocessing the data.
    """

    tokenizing.prepare_tokenizer_data.prepare_tokenizer_data()
    tokenizing.convert_to_parquet.convert_to_parquet()
    tokenizing.train_tokenizer.train_tokenizer()